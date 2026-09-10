"""Real GNU Radio smoke tests. No hardware, network or subscriber data needed."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from hackrf_simple import Reporter, build_decoder, make_parser, recording_settings
from test_hackrf_simple import IMSI, packet


AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("gnuradio", "grgsm", "pmt"))


@unittest.skipUnless(AVAILABLE, "Requires GNU Radio and gr-gsm (run in Debian CI)")
class IntegrationTests(unittest.TestCase):
    def test_real_pmt_to_observation(self):
        import pmt
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zero.cs8"
            path.write_bytes(b"\0" * 200000)
            args = make_parser().parse_args(["decode", str(path), "-f", "935.2M", "-s", "8M", "--format", "cs8"])
            reporter = Reporter(args, 935.2e6)
            graph = build_decoder(args, recording_settings(args), reporter)
            data = packet(b"\x06\x21\x00\x08" + IMSI)
            message = pmt.cons(pmt.PMT_NIL, pmt.init_u8vector(len(data), list(data)))
            with redirect_stdout(io.StringIO()) as output:
                graph.packet_sink.receive(message)
            self.assertIsNone(reporter.error)
            self.assertIn("001070000000001", output.getvalue())

    def test_finite_cs8_and_cf32_graphs_finish(self):
        script = Path(__file__).resolve().parents[1] / "hackrf_simple.py"
        with tempfile.TemporaryDirectory() as directory:
            for fmt, width, mode in (("cs8", 2, "BCCH"), ("cf32", 8, "BCCH_SDCCH4")):
                path = Path(directory) / ("zero." + fmt)
                path.write_bytes(b"\0" * (100000 * width))
                result = subprocess.run([sys.executable, str(script), "decode", str(path),
                                         "-f", "935.2M", "-s", "8M", "--format", fmt, "--mode", mode],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Finished: 0 observations", result.stderr)


if __name__ == "__main__":
    unittest.main()
