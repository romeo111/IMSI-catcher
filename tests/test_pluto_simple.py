from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hackrf_simple import recording_settings, sidecar
from pluto_simple import capture, make_parser, validate_capture

try:
    import numpy as np
except ImportError:
    np = None


class FakeContext:
    def set_timeout(self, value):
        self.timeout = value


class FakePluto:
    """One RX, rounded rate, finite buffers. Any direct TX use fails the test."""
    def __init__(self, uri):
        self.uri = uri
        self.ctx = FakeContext()
        self.calls = 0
        self.destroyed = False

    def __setattr__(self, name, value):
        if name.startswith("tx") or "chan1" in name:
            raise AssertionError("Only RX0 is supported")
        object.__setattr__(self, name, value)

    @property
    def sample_rate(self):
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value):
        self._sample_rate = value + 1  # emulate synthesizer rounding

    def rx(self):
        self.calls += 1
        return np.arange(4, dtype=float) + self.calls * 1000 + 1j * np.arange(4)

    def rx_destroy_buffer(self):
        self.destroyed = True


class ValidationTests(unittest.TestCase):
    def test_pluto_defaults_and_decoder_compatibility(self):
        args = make_parser().parse_args(["capture", "sample.cf32", "-f", "935.2M"])
        validate_capture(args)
        self.assertEqual((args.sample_rate, args.bandwidth, args.uri),
                         (2e6, 600e3, "ip:192.168.2.1"))
        args = make_parser().parse_args(["decode", "sample.cf32", "--all-tmsi", "--assignments"])
        self.assertTrue(args.all_tmsi and args.assignments)

    def test_invalid_settings(self):
        for name, value in (("freq", 70e6), ("freq", 6e9), ("sample_rate", 200e3),
                            ("bandwidth", 3e6), ("gain", float("nan")), ("gain", 71),
                            ("buffer_size", 0), ("timeout_ms", 0), ("uri", " ")):
            args = make_parser().parse_args(["capture", "sample.cf32", "-f", "935.2M"])
            setattr(args, name, value)
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                validate_capture(args)


@unittest.skipUnless(np is not None, "Requires NumPy")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.cf32"
        self.args = make_parser().parse_args(["capture", str(self.path), "-f", "935.2M",
                                             "--seconds", "0.000003", "--uri", "ip:test-device"])
        self.device = FakePluto(uri=self.args.uri)

    def run_capture(self):
        with redirect_stderr(io.StringIO()):
            capture(self.args, device_factory=lambda uri: self.device)

    def test_exact_count_iq_order_readback_and_metadata(self):
        self.run_capture()
        data = np.fromfile(self.path, dtype="<c8")
        expected = [2000, 2001 + 1j, 2002 + 2j, 2003 + 3j, 3000, 3001 + 1j]
        np.testing.assert_array_equal(data, expected)
        metadata = json.loads(sidecar(self.path).read_text())
        self.assertEqual((metadata["format"], metadata["status"], metadata["samples"], metadata["bytes"]),
                         ("cf32", "complete", 6, 48))
        self.assertEqual(metadata["sample_rate"], 2000001)
        self.assertEqual(metadata["rx_channel"], 0)
        self.assertEqual(self.device.rx_enabled_channels, [0])
        self.assertEqual(self.device.ctx.timeout, 5000)
        self.assertTrue(self.device.destroyed)
        args = make_parser().parse_args(["decode", str(self.path)])
        self.assertEqual(recording_settings(args), (935.2e6, 2000001, "cf32"))

    def test_timeout_preserves_partial_recording(self):
        original = self.device.rx
        def receive():
            if self.device.calls == 2:
                raise TimeoutError("simulated timeout")
            return original()
        self.device.rx = receive
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.run_capture()
        metadata = json.loads(sidecar(self.path).read_text())
        self.assertEqual((metadata["status"], metadata["samples"]), ("partial", 4))
        self.assertEqual(self.path.stat().st_size, 32)
        self.assertTrue(self.device.destroyed)

    def test_ctrl_c_preserves_partial_recording(self):
        self.device.rx = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            self.run_capture()
        self.assertEqual(json.loads(sidecar(self.path).read_text())["status"], "partial")
        self.assertTrue(self.device.destroyed)

    def test_existing_output_is_untouched(self):
        self.path.write_bytes(b"existing")
        with self.assertRaises(ValueError):
            self.run_capture()
        self.assertEqual(self.path.read_bytes(), b"existing")
        self.assertFalse(sidecar(self.path).exists())
        self.assertEqual(self.device.calls, 0)

    def test_reject_multichannel_empty_and_real_arrays(self):
        for bad in (np.zeros((2, 4), complex), np.array([], complex), np.zeros(4),
                    np.array([complex(float("nan"), 0)])):
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                self.device.rx = lambda: bad
                with self.assertRaises(RuntimeError):
                    self.run_capture()
                self.assertEqual(json.loads(sidecar(self.path).read_text())["status"], "partial")
                self.path.unlink()
                sidecar(self.path).unlink()

    def test_connection_error_is_actionable(self):
        with self.assertRaisesRegex(RuntimeError, "Check URI"):
            capture(self.args, device_factory=lambda uri: (_ for _ in ()).throw(OSError("unreachable")))
        self.assertEqual(json.loads(sidecar(self.path).read_text())["status"], "partial")

    def test_cleanup_error_does_not_discard_completed_capture(self):
        self.device.rx_destroy_buffer = lambda: (_ for _ in ()).throw(OSError("device disconnected"))
        self.run_capture()
        metadata = json.loads(sidecar(self.path).read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertIn("cleanup_error", metadata)


@unittest.skipUnless(importlib.util.find_spec("adi") is not None, "Requires pyadi-iio")
class DriverApiTests(unittest.TestCase):
    def test_installed_pluto_driver_has_required_rx_api(self):
        # Import real pyadi/libiio without discovering or opening a radio.
        from adi import Pluto
        for name in ("rx", "rx_destroy_buffer", "sample_rate", "rx_lo", "rx_rf_bandwidth",
                     "rx_enabled_channels", "rx_annotated", "rx_output_type", "rx_buffer_size",
                     "gain_control_mode_chan0", "rx_hardwaregain_chan0", "ctx"):
            self.assertTrue(hasattr(Pluto, name), name)


if __name__ == "__main__":
    unittest.main()
