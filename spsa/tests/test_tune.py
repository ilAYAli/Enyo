#!/usr/bin/env python3

import fcntl
import json
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tune.py"


def executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SpsaTuneTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-tune-")
        self.root = Path(self.temporary.name)
        self.params = self.root / "params.txt"
        self.state = self.root / "state.json"
        self.fastchess = self.root / "fastchess"
        self.params.write_text("alpha, 1, 0, 10, 1, 0.03\n")
        executable(
            self.fastchess,
            "#!/bin/sh\n"
            "echo 'Games: 2, Wins: 1, Losses: 0, Draws: 1'\n",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def command(self):
        return [
            str(SCRIPT),
            "--params", str(self.params),
            "--state", str(self.state),
            "--fastchess", str(self.fastchess),
            "--iterations", "1",
            "--rounds", "1",
            "--concurrency", "1",
            "--seed", "1",
        ]

    def test_writes_state_and_csv_next_to_each_other(self):
        result = subprocess.run(
            self.command(), check=True, capture_output=True, text=True
        )

        state = json.loads(self.state.read_text())
        self.assertEqual(state["k"], 1)
        self.assertEqual(state["names"], ["alpha"])
        self.assertTrue(self.state.with_suffix(".csv").is_file())
        self.assertIn("final values:", result.stdout)

    def test_rejects_a_second_tuner_for_the_same_state(self):
        lock_path = self.state.with_suffix(".lock")
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run(
                self.command(), check=False, capture_output=True, text=True
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("another SPSA tuner is already using", result.stderr)
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
