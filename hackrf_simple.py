#!/usr/bin/env python3
"""HackRF capture and single-carrier GSM analysis. CC0-1.0, see LICENSE."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

from hackrf_packets import parse_packet


def number(value):
    text = str(value).strip().lower()
    multiplier = {"k": 1e3, "m": 1e6, "g": 1e9}.get(text[-1:] or "")
    try:
        result = float(text[:-1] if multiplier else text) * (multiplier or 1)
    except ValueError:
        raise argparse.ArgumentTypeError("Use a number such as 8M or 935.2M")
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("Value must be positive and finite")
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sidecar(path):
    return Path(str(path) + ".json")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check installed dependencies; does not open the radio")
    capture = commands.add_parser("capture", help="Record signed 8-bit I/Q with metadata")
    capture.add_argument("output", type=Path)
    capture.add_argument("-f", "--freq", type=number, required=True, help="Carrier frequency in Hz, e.g. 935.2M")
    capture.add_argument("-s", "--sample-rate", type=number, default=8e6)
    capture.add_argument("--seconds", type=number, default=30)
    capture.add_argument("--lna", type=int, choices=range(0, 41, 8), default=16)
    capture.add_argument("--vga", type=int, choices=range(0, 63, 2), default=20)
    capture.add_argument("--serial", help="HackRF serial number when multiple devices are connected")
    decode = commands.add_parser("decode", help="Analyze a saved IQ file (metadata loaded automatically)")
    decode.add_argument("input", type=Path)
    decode.add_argument("-f", "--freq", type=number, help="Recorded center frequency; override metadata")
    decode.add_argument("-s", "--sample-rate", type=number, help="Actual recorded sample rate")
    decode.add_argument("--format", choices=("cs8", "cf32"), help="Raw signed int8 I/Q or little-endian float32 I/Q")
    decode.add_argument("--ppm", type=float, default=0, help="Software frequency correction")
    decode.add_argument("--mode", choices=("BCCH", "BCCH_SDCCH4"), default="BCCH")
    decode.add_argument("--all-tmsi", action="store_true", help="Also display temporary identities")
    decode.add_argument("--assignments", action="store_true", help="Also display Immediate Assignment messages")
    decode.add_argument("--jsonl", type=Path, help="Write observations to a NEW JSON Lines file")
    return parser


def capture_command(args, executable="hackrf_transfer"):
    if not 8e6 <= args.sample_rate <= 20e6:
        raise ValueError("HackRF capture sample rate must be 8M..20M")
    if not 1e6 <= args.freq <= 6e9:
        raise ValueError("Frequency must be between 1M and 6G")
    if not args.sample_rate.is_integer() or not args.freq.is_integer():
        raise ValueError("Frequency and sample rate must be whole Hz")
    samples = int(args.sample_rate * args.seconds)
    if not 1 <= samples < 2**63:
        raise ValueError("Requested sample count is out of range")
    command = [executable, "-r", str(args.output), "-f", str(int(args.freq)),
               "-s", str(int(args.sample_rate)), "-n", str(samples),
               "-b", "1750000", "-l", str(args.lna), "-g", str(args.vga), "-a", "0"]
    if args.serial:
        command.extend(("-d", args.serial))
    return command


def capture(args):
    command = capture_command(args)
    if args.output.exists() or sidecar(args.output).exists():
        raise ValueError("Output or metadata already exists; choose a new filename")
    executable = shutil.which(command[0])
    if not executable:
        raise RuntimeError("hackrf_transfer is missing. Install the 'hackrf' package.")
    command[0] = executable
    metadata = {"format": "cs8", "sample_rate": args.sample_rate,
                "center_frequency": args.freq, "started_utc": utc_now(),
                "requested_seconds": args.seconds, "lna": args.lna, "vga": args.vga,
                "bandwidth": 1750000, "status": "recording"}
    # Exclusive create prevents accidentally replacing previous metadata.
    with sidecar(args.output).open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print("Recording at {:g} MS/s, about {:.1f} MB.".format(
        args.sample_rate / 1e6, args.sample_rate * args.seconds * 2 / 1e6), file=sys.stderr)
    code = None
    try:
        code = subprocess.run(command, check=False).returncode
    finally:
        size = args.output.stat().st_size if args.output.exists() else 0
        expected_size = int(args.sample_rate * args.seconds) * 2
        metadata.update(status="complete" if code == 0 and size == expected_size else "partial",
                        bytes=size, recorded_seconds=size / (2 * args.sample_rate),
                        finished_utc=utc_now())
        sidecar(args.output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if code != 0:
        raise RuntimeError("hackrf_transfer failed (exit {}). Any partial recording was kept.".format(code))
    if metadata["status"] != "complete":
        raise RuntimeError("Capture ended early. Partial recording and metadata were kept.")


def recording_settings(args):
    if not args.input.is_file():
        raise ValueError("IQ file does not exist: {}".format(args.input))
    metadata = {}
    if sidecar(args.input).exists():
        metadata = json.loads(sidecar(args.input).read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("IQ metadata must be a JSON object")
    frequency = args.freq if args.freq is not None else metadata.get("center_frequency")
    rate = args.sample_rate if args.sample_rate is not None else metadata.get("sample_rate")
    fmt = args.format or metadata.get("format")
    if frequency is None or rate is None or fmt is None:
        raise ValueError("Without metadata, specify --freq, --sample-rate and --format")
    try:
        frequency, rate = float(frequency), float(rate)
    except (TypeError, ValueError):
        raise ValueError("Metadata frequency and sample rate must be numbers")
    if not math.isfinite(frequency) or not 1e6 <= frequency <= 6e9:
        raise ValueError("Invalid recorded center frequency")
    if not math.isfinite(rate) or not 250e3 <= rate <= 20e6:
        raise ValueError("Recorded sample rate must be 250k..20M")
    if fmt not in ("cs8", "cf32"):
        raise ValueError("Supported raw formats: cs8 and cf32")
    if not math.isfinite(args.ppm):
        raise ValueError("PPM must be finite")
    width = 2 if fmt == "cs8" else 8
    size = args.input.stat().st_size
    if size == 0 or size % width:
        raise ValueError("IQ file is empty or has an incomplete I/Q sample")
    with args.input.open("rb") as handle:
        if handle.read(4) in (b"RIFF", b"RF64"):
            raise ValueError("WAV containers are unsupported; export raw I/Q first")
    return frequency, rate, fmt


class Reporter:
    def __init__(self, args, frequency, output=None):
        self.args, self.frequency, self.output = args, frequency, output
        self.cell = {}
        self.count = 0
        self.error = None

    def handle(self, packet):
        for event in parse_packet(packet):
            if event["event"] == "cell":
                new_cell = {key: event[key] for key in ("mcc", "mnc", "lac", "cell")}
                if new_cell == self.cell:
                    continue
                self.cell = new_cell
            if event["event"] == "identity" and event["identity_type"] == "TMSI" and not self.args.all_tmsi:
                continue
            if event["event"] == "assignment" and not self.args.assignments:
                continue
            row = dict(self.cell, **{key: value for key, value in event.items() if key not in self.cell})
            row.update(processed_utc=utc_now(), center_frequency=self.frequency)
            text = " ".join("{}={}".format(key, value) for key, value in row.items()
                            if key != "processed_utc" and value is not None)
            print(text, flush=True)
            if self.output:
                self.output.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.output.flush()
            self.count += 1


def build_decoder(args, settings, reporter):
    """Construct a file-only flowgraph; no UDP port or radio device is opened."""
    try:
        from gnuradio import blocks, gr
        import grgsm
        import pmt
    except ImportError as error:
        raise RuntimeError("Use Linux system Python with 'gnuradio' and 'gr-gsm' installed: {}".format(error))
    frequency, rate, fmt = settings
    if sys.byteorder != "little" and fmt == "cf32":
        raise RuntimeError("cf32 input requires a little-endian host")

    class PacketSink(gr.basic_block):
        def __init__(self):
            gr.basic_block.__init__(self, name="GSM observations", in_sig=None, out_sig=None)
            self.message_port_register_in(pmt.intern("in"))
            self.set_msg_handler(pmt.intern("in"), self.receive)

        def receive(self, message):
            try:
                data = pmt.cdr(message)
                # PMT blobs use the u8vector representation. blob_data() exposes
                # a C pointer/PyCapsule, so it must not be converted with bytes().
                if pmt.is_u8vector(data):
                    packet = bytes(pmt.u8vector_elements(data))
                else:
                    return
                reporter.handle(packet)
            except Exception as error:
                # Surface asynchronous output/parser errors after the graph stops.
                reporter.error = error

    graph = gr.top_block("HackRF IQ GSM decoder")
    source = blocks.file_source(gr.sizeof_char if fmt == "cs8" else gr.sizeof_gr_complex,
                                str(args.input.resolve()), False)
    adapter = grgsm.gsm_input(ppm=args.ppm, osr=4, fc=frequency, samp_rate_in=rate)
    if fmt == "cs8":
        converter = blocks.interleaved_char_to_complex()
        scale = blocks.multiply_const_cc(1.0 / 128.0)
        graph.connect(source, converter, scale, adapter)
    else:
        graph.connect(source, adapter)
    arfcn = grgsm.arfcn.downlink2arfcn(frequency)
    if arfcn is None or arfcn < 0:
        raise ValueError("Center frequency must be a GSM downlink carrier")
    receiver = grgsm.receiver(4, [arfcn], [])
    correction = grgsm.clock_offset_control(frequency, rate)
    demapper = (grgsm.gsm_bcch_ccch_demapper(0) if args.mode == "BCCH"
                else grgsm.gsm_bcch_ccch_sdcch4_demapper(0))
    decoder = grgsm.control_channels_decoder()
    sink = PacketSink()
    graph.connect(adapter, receiver)
    graph.msg_connect(receiver, "measurements", correction, "measurements")
    graph.msg_connect(correction, "ctrl", adapter, "ctrl_in")
    graph.msg_connect(receiver, "C0", demapper, "bursts")
    graph.msg_connect(demapper, "bursts", decoder, "bursts")
    graph.msg_connect(decoder, "msgs", sink, "in")
    graph.packet_sink = sink
    return graph


def decode(args):
    settings = recording_settings(args)
    if args.jsonl and (args.jsonl.exists() or args.jsonl.resolve() in
                       (args.input.resolve(), sidecar(args.input).resolve())):
        raise ValueError("JSONL output already exists or conflicts with the input")
    with ExitStack() as stack:
        reporter = Reporter(args, settings[0])
        graph = build_decoder(args, settings, reporter)
        if args.jsonl:
            reporter.output = stack.enter_context(args.jsonl.open("x", encoding="utf-8"))
        print("Decoding {:g} MHz at {:g} MS/s ({})".format(
            settings[0] / 1e6, settings[1] / 1e6, settings[2]), file=sys.stderr)
        try:
            graph.start()
            graph.wait()
        finally:
            graph.stop()
            graph.wait()
        if reporter.error:
            raise RuntimeError("Output processing failed: {}".format(reporter.error))
        print("Finished: {} observations.{}".format(reporter.count,
              " Check frequency, sample format, PPM and signal quality if no cells appeared."
              if not reporter.cell else ""), file=sys.stderr)


def doctor():
    missing = []
    for name in ("gnuradio", "grgsm", "pmt"):
        present = importlib.util.find_spec(name) is not None
        print("{}: {}".format(name, "found" if present else "missing"))
        if not present:
            missing.append(name)
    present = shutil.which("hackrf_transfer") is not None
    print("hackrf_transfer: {}".format("found" if present else "missing (needed only for capture)"))
    if not present:
        missing.append("hackrf_transfer")
    print("Use Linux /usr/bin/python3. Hardware is not checked by this command.")
    return 1 if missing else 0


def main(argv=None):
    parser = make_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return doctor()
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
