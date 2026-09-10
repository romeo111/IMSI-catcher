#!/usr/bin/env python3
"""Single-RX Pluto / Nano SDR IQ capture and offline GSM analysis. CC0-1.0."""

import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import time

from hackrf_simple import add_decode_parser, decode, load_gsm, number, sidecar, utc_now


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("doctor", help="Check software dependencies without connecting to a radio")
    check.add_argument("--capture-only", action="store_true", help="Skip Linux GNU Radio checks")
    capture = commands.add_parser("capture", help="Record RX0 as raw cf32 with metadata")
    capture.add_argument("output", type=Path)
    capture.add_argument("--uri", default="ip:192.168.2.1", help="libiio URI, e.g. ip:192.168.2.1 or usb:1.2.5")
    capture.add_argument("-f", "--freq", type=number, required=True, help="Center frequency of GSM C0, e.g. 935.2M")
    capture.add_argument("-s", "--sample-rate", type=number, default=2e6)
    capture.add_argument("--bandwidth", type=number, default=600e3)
    capture.add_argument("--seconds", type=number, default=30)
    capture.add_argument("--gain", type=float, default=40, help="Manual RX0 gain in dB (default 40)")
    capture.add_argument("--buffer-size", type=int, default=65536, help="Complex samples per RX buffer")
    capture.add_argument("--timeout-ms", type=int, default=5000, help="libiio receive timeout")
    add_decode_parser(commands)
    return parser


def validate_capture(args):
    # Deliberately use the standard AD9363 tuning range, without firmware mods.
    if not 325e6 <= args.freq <= 3.8e9:
        raise ValueError("Standard AD9363 frequency range is 325M..3.8G")
    if not 1e6 <= args.sample_rate <= 20e6:
        raise ValueError("This single-carrier recorder supports 1M..20M sample rates")
    if not 200e3 <= args.bandwidth <= args.sample_rate:
        raise ValueError("Bandwidth must be at least 200k and no greater than sample rate")
    if any(not float(value).is_integer() for value in (args.freq, args.sample_rate, args.bandwidth)):
        raise ValueError("Frequency, sample rate and bandwidth must be whole Hz")
    if not math.isfinite(args.gain) or not 0 <= args.gain <= 70:
        raise ValueError("Manual RX gain must be 0..70 dB")
    if not 1024 <= args.buffer_size <= 1048576:
        raise ValueError("Buffer size must be 1024..1048576 complex samples")
    if not 100 <= args.timeout_ms <= 60000:
        raise ValueError("Timeout must be 100..60000 ms")
    if not args.uri.strip():
        raise ValueError("libiio URI cannot be empty")
    if not 1 <= args.sample_rate * args.seconds < 2**63:
        raise ValueError("Requested sample count is out of range")


def capture(args, device_factory=None):
    validate_capture(args)
    if args.output.exists() or sidecar(args.output).exists():
        raise ValueError("Output or metadata already exists; choose a new filename")
    try:
        import numpy as np
        if device_factory is None:
            from adi import Pluto
            device_factory = Pluto
    except (ImportError, OSError) as error:
        raise RuntimeError("Install NumPy, pyadi-iio and native libiio; see PLUTO_UK.md: {}".format(error))

    device = None
    saved = 0
    rate = None
    complete = False
    metadata = {"device": "pluto-compatible", "rx_channel": 0, "uri": args.uri,
                "format": "cf32", "sample_units": "raw_adc_counts",
                "requested_frequency": args.freq, "requested_sample_rate": args.sample_rate,
                "requested_seconds": args.seconds, "requested_bandwidth": args.bandwidth,
                "requested_gain_db": args.gain, "status": "initializing",
                "continuity": "not_verified"}
    with sidecar(args.output).open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    started = time.monotonic()
    try:
        with args.output.open("xb") as output:
            try:
                device = device_factory(uri=args.uri)
                device.ctx.set_timeout(args.timeout_ms)
                device.rx_enabled_channels = [0]
                device.rx_annotated = False
                device.rx_output_type = "raw"
                device.sample_rate = int(args.sample_rate)
                device.rx_rf_bandwidth = int(args.bandwidth)
                device.rx_lo = int(args.freq)
                device.gain_control_mode_chan0 = "manual"
                device.rx_hardwaregain_chan0 = args.gain
                device.rx_buffer_size = args.buffer_size
                rate = int(device.sample_rate)
                frequency = int(device.rx_lo)
                bandwidth = int(device.rx_rf_bandwidth)
                if not 1e6 <= rate <= 20e6 or not 325e6 <= frequency <= 3.8e9:
                    raise ValueError("Device returned unsupported frequency/sample rate")
                if list(device.rx_enabled_channels) != [0]:
                    raise ValueError("Device did not select exactly RX0")
                samples = int(rate * args.seconds)
                if not 1 <= samples < 2**63:
                    raise ValueError("Read-back sample rate produces an invalid sample count")
                metadata.update(sample_rate=rate, center_frequency=frequency,
                                bandwidth=bandwidth, gain_db=float(device.rx_hardwaregain_chan0),
                                gain_control_mode=str(device.gain_control_mode_chan0),
                                buffer_size=args.buffer_size)
            except Exception as error:
                raise RuntimeError("Cannot configure Pluto RX at {}: {}. Check URI, Pluto firmware and libiio.".format(
                    args.uri, error)) from error

            print("RX0: {:g} MHz, {:g} MS/s, {:g} kHz bandwidth; about {:.1f} MB.".format(
                frequency / 1e6, rate / 1e6, bandwidth / 1e3, samples * 8 / 1e6), file=sys.stderr)
            # Discard one startup buffer after tuning/gain calibration.
            device.rx()
            metadata.update(started_utc=utc_now(), status="recording")
            sidecar(args.output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            started = time.monotonic()
            while saved < samples:
                data = np.asarray(device.rx())
                if data.ndim != 1 or data.size == 0 or not np.iscomplexobj(data):
                    raise RuntimeError("Expected one nonempty complex RX0 array from Pluto")
                # Preserve ADC counts and I/Q order; no HackRF int8 quantization.
                block = data[:samples - saved].astype("<c8")
                if not np.isfinite(block).all():
                    raise RuntimeError("Device returned non-finite IQ values")
                output.write(block.tobytes())
                saved += block.size
            complete = True
    except Exception as error:
        metadata["error"] = str(error)
        raise RuntimeError("Pluto capture failed; partial files were kept: {}".format(error)) from error
    finally:
        elapsed = time.monotonic() - started
        metadata.update(status="complete" if complete else "partial", samples=saved,
                        bytes=saved * 8, recorded_seconds=saved / rate if rate else 0,
                        elapsed_seconds=elapsed, finished_utc=utc_now())
        # Mark a hardware I/O timeout or Ctrl+C as partial and release DMA buffers.
        try:
            if device is not None:
                try:
                    device.rx_destroy_buffer()
                except Exception as error:
                    metadata["cleanup_error"] = str(error)
                    print("Could not release RX buffer: {}".format(error), file=sys.stderr)
        finally:
            sidecar(args.output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved {} and {}".format(args.output, sidecar(args.output)), file=sys.stderr)


def doctor(capture_only=False):
    failed = False
    names = ["numpy", "iio", "adi"]
    if not capture_only:
        names.extend(("gnuradio", "grgsm", "pmt"))
    for name in names:
        try:
            module = load_gsm() if name == "grgsm" else importlib.import_module(name)
            if name == "adi" and not hasattr(module, "Pluto"):
                raise ImportError("The 'adi' module must come from pyadi-iio")
            print("{}: found ({})".format(name, module.__name__))
        except (ImportError, OSError) as error:
            print("{}: unavailable ({})".format(name, error))
            failed = True
    print("Hardware was not opened. Pluto-compatible libiio firmware is required; see PLUTO_UK.md.")
    return int(failed)


def main(argv=None):
    args = make_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor(args.capture_only)
        if args.command == "capture":
            capture(args)
        else:
            decode(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print("Error: {}".format(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
