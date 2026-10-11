#!/usr/bin/env python3
"""Regression test: child training output must reach both log sinks immediately."""

import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vkong_transfer_service import stream_process_output


class TrainingLogStreamTest(unittest.TestCase):
    def test_streams_stdout_and_stderr_bytes_to_file_and_console(self):
        expected = b"step 1 loss=0.5\rwarning from stderr\n"
        child_code = (
            "import os; os.write(1, b'step 1 loss=0.5\\r'); "
            "os.write(2, b'warning from stderr\\n')"
        )
        with tempfile.TemporaryFile() as log:
            console = io.BytesIO()
            process = subprocess.Popen(
                [sys.executable, "-c", child_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            exit_code = stream_process_output(process, log, console)
            log.seek(0)
            self.assertEqual(exit_code, 0)
            self.assertEqual(log.read(), expected)
            self.assertEqual(console.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
