import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

from hackrf_packets import mobile_identity, parse_packet
from hackrf_simple import (Reporter, capture, capture_command, make_parser,
                           number, recording_settings, sidecar)


def packet(data, subtype=2, extension=b"", arfcn=1):
    assert len(extension) % 4 == 0
    header = struct.pack("!BBBBHbbIBBBB", 2, 4 + len(extension) // 4,
                         1, 0, arfcn, -40, 20, 1234, subtype, 0, 0, 0)
    return header + extension + bytes([(len(data) << 2) | 1]) + data


# Synthetic test-network IMSI, not a captured subscriber identity.
IMSI = bytes.fromhex("09 10 70 00 00 00 00 10")


class PacketTests(unittest.TestCase):
    def test_imsi_digits_and_even_filler(self):
        self.assertEqual(mobile_identity(IMSI)["identity"], "001070000000001")
        self.assertEqual(mobile_identity(bytes.fromhex("01 10 70 f0"))["identity"], "001070")
        for invalid in (b"", b"\x09", b"\x03" + IMSI[1:], b"\x09\xff\xff", b"\x01\x00\x00"):
            self.assertIsNone(mobile_identity(invalid))

    def test_paging_separate_identities(self):
        data = b"\x06\x21\x00\x08" + IMSI + bytes.fromhex("17 05 f4 12 34 56 78")
        events = parse_packet(packet(data))
        self.assertEqual([e["identity_type"] for e in events], ["IMSI", "TMSI"])
        self.assertEqual(events[1]["identity"], "12345678")
        self.assertNotIn("imsi", events[1])

    def test_two_imsi_paging(self):
        data = b"\x06\x21\x00\x08" + IMSI + b"\x17\x08" + IMSI
        self.assertEqual(len(parse_packet(packet(data))), 2)

    def test_paging_types_two_three(self):
        data = bytes.fromhex("06 22 00 00 00 00 01 00 00 00 02 17 08") + IMSI
        self.assertEqual(len(parse_packet(packet(data))), 3)
        data = bytes.fromhex("06 24 00") + bytes(range(16))
        self.assertEqual(len(parse_packet(packet(data))), 4)

    def test_cell_two_and_three_digit_mnc(self):
        data = bytes.fromhex("06 1b 61 9d 00 f1 70 01 9c")
        cell = parse_packet(packet(data, 1))[0]
        self.assertEqual((cell["mcc"], cell["mnc"], cell["cell"], cell["lac"]),
                         ("001", "07", 24989, 412))
        data = bytes.fromhex("06 1b 61 9d 00 51 00 01 9c")
        self.assertEqual(parse_packet(packet(data, 1))[0]["mnc"], "005")

    def test_assignment_and_hopping(self):
        data = bytes.fromhex("06 3f 00 79 03 df 00 00 00 00 00")
        event = parse_packet(packet(data))[0]
        self.assertEqual((event["channel"], event["subchannel"], event["assigned_timeslot"], event["assigned_arfcn"]),
                         ("SDCCH/8", 7, 1, 991))
        data = data[:4] + b"\x13" + data[5:]
        self.assertTrue(parse_packet(packet(data))[0]["hopping"])
        self.assertIsNone(parse_packet(packet(data))[0]["assigned_arfcn"])

    def test_header_extensions_and_truncation(self):
        data = b"\x06\x21\x00\x08" + IMSI
        p = packet(data, extension=b"\0" * 4)
        self.assertEqual(parse_packet(p)[0]["identity"], "001070000000001")
        for end in range(len(p)):
            self.assertEqual(parse_packet(p[:end]), [])

    def test_reject_unsupported_framing(self):
        data = b"\x06\x21\x00\x08" + IMSI
        self.assertEqual(parse_packet(packet(data, subtype=8)), [])
        self.assertEqual(parse_packet(packet(data, arfcn=0x4001)), [])
        p = bytearray(packet(data))
        p[1] = 3
        self.assertEqual(parse_packet(p), [])

    def test_random_short_packets_do_not_crash(self):
        rng = random.Random(17)
        for _ in range(2000):
            parse_packet(bytes(rng.randrange(256) for _ in range(rng.randrange(128))))


class CliTests(unittest.TestCase):
    def test_frequency_numbers(self):
        self.assertEqual(number("935.2M"), 935200000)
        for value in ("nan", "inf", "-1", "0", "junk", ""):
            with self.assertRaises(argparse.ArgumentTypeError):
                number(value)

    def test_receive_only_capture_defaults_and_validation(self):
        args = make_parser().parse_args(["capture", "test.cs8", "-f", "935.2M"])
        command = capture_command(args)
        self.assertIn("8000000", command)
        self.assertIn("240000000", command)
        self.assertNotIn("-t", command)
        args.sample_rate = 2e6
        with self.assertRaises(ValueError):
            capture_command(args)

    def test_metadata_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "example.cs8"
            path.write_bytes(b"\0" * 16)
            sidecar(path).write_text(json.dumps({"center_frequency": 935.2e6,
                                               "sample_rate": 8e6, "format": "cs8"}))
            args = make_parser().parse_args(["decode", str(path)])
            self.assertEqual(recording_settings(args), (935.2e6, 8e6, "cs8"))
            args.sample_rate = 2e6
            self.assertEqual(recording_settings(args)[1], 2e6)
            path.write_bytes(b"\0" * 3)
            with self.assertRaises(ValueError):
                recording_settings(args)

    def test_raw_requires_metadata_and_rejects_wav(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording.iq"
            path.write_bytes(b"RIFF" + b"\0" * 12)
            args = make_parser().parse_args(["decode", str(path)])
            with self.assertRaises(ValueError):
                recording_settings(args)
            args.freq, args.sample_rate, args.format = 935.2e6, 8e6, "cs8"
            with self.assertRaisesRegex(ValueError, "WAV"):
                recording_settings(args)

    def test_capture_failure_preserves_partial_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "capture.cs8"
            args = make_parser().parse_args(["capture", str(path), "-f", "935.2M"])
            def fail(*unused, **kwargs):
                path.write_bytes(b"\0\0")
                return argparse.Namespace(returncode=1)
            with patch("hackrf_simple.shutil.which", return_value="fake-hackrf"), \
                 patch("hackrf_simple.subprocess.run", side_effect=fail):
                with self.assertRaises(RuntimeError):
                    capture(args)
            self.assertEqual(json.loads(sidecar(path).read_text())["status"], "partial")
            with self.assertRaises(ValueError):
                capture(args)

    def test_reporter_json_and_default_tmsi_filter(self):
        args = make_parser().parse_args(["decode", "unused"])
        output = io.StringIO()
        reporter = Reporter(args, 935.2e6, output)
        data = b"\x06\x21\x00\x08" + IMSI + bytes.fromhex("17 05 f4 12 34 56 78")
        with redirect_stdout(io.StringIO()):
            reporter.handle(packet(data))
        row = json.loads(output.getvalue())
        self.assertEqual(row["identity_type"], "IMSI")
        self.assertIn("processed_utc", row)
        self.assertNotIn("captured_utc", row)


if __name__ == "__main__":
    unittest.main()
