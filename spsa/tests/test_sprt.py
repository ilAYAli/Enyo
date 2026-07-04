#!/usr/bin/env python3

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "sprt.py"


def executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class SpsaSprtTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-sprt-")
        self.root = Path(self.temporary.name)
        self.home = self.root / "home"
        self.home.mkdir()
        self.state = self.root / "state.json"
        self.params = self.root / "params.txt"
        self.engine = self.root / "engine"
        self.net = self.root / "network.nn"
        self.book = self.root / "openings.epd"
        self.forge = self.root / "forge"
        self.capture = self.root / "capture.json"
        self.params.write_text(
            "alpha, 1, 0, 10, 1, 0.03\n"
            "beta, 2, 0, 10, 1, 0.03\n"
        )
        self.state.write_text(json.dumps({
            "k": 1500,
            "names": ["alpha", "beta"],
            "theta": [1.6, 3.6],
        }))
        executable(self.engine, "#!/bin/sh\nexit 0\n")
        self.net.write_text("network\n")
        self.book.write_text("openings\n")
        executable(self.forge, """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
Path(os.environ["CAPTURE"]).write_text(json.dumps({
    "args": sys.argv[1:],
    "cwd": os.getcwd(),
}))
""")

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *extra: str):
        return [
            str(SCRIPT),
            "--state", str(self.state),
            "--params", str(self.params),
            "--engine", str(self.engine),
            "--net", str(self.net),
            "--book", str(self.book),
            "--forge", str(self.forge),
            "--run", "spsa-test",
            *extra,
        ]

    def environment(self):
        environment = os.environ.copy()
        environment.update({"HOME": str(self.home), "CAPTURE": str(self.capture)})
        return environment

    def test_runs_forge_from_home_with_default_and_tuned_options(self):
        result = subprocess.run(
            self.command(),
            cwd=self.root,
            env=self.environment(),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Starting Forge run spsa-test", result.stdout)
        capture = json.loads(self.capture.read_text())
        self.assertEqual(Path(capture["cwd"]).resolve(), self.home.resolve())
        arguments = capture["args"]
        self.assertEqual(arguments[0], "sprt")
        self.assertNotIn("--wait", arguments)
        self.assertEqual(arguments[arguments.index("--games") + 1], "1000")
        reference = [
            arguments[index + 1]
            for index, value in enumerate(arguments)
            if value == "--reference-uci"
        ]
        candidate = [
            arguments[index + 1]
            for index, value in enumerate(arguments)
            if value == "--candidate-uci"
        ]
        self.assertIn("alpha=1", reference)
        self.assertIn("beta=2", reference)
        self.assertIn("alpha=2", candidate)
        self.assertIn("beta=4", candidate)
        self.assertIn(f"nnue_file={self.net}", reference)
        self.assertIn(f"nnue_file={self.net}", candidate)

    def test_wait_is_explicit(self):
        subprocess.run(
            self.command("--wait"),
            cwd=self.root,
            env=self.environment(),
            check=True,
            capture_output=True,
            text=True,
        )

        arguments = json.loads(self.capture.read_text())["args"]
        self.assertIn("--wait", arguments)

    def test_dry_run_prints_command_without_starting_forge(self):
        result = subprocess.run(
            self.command("--dry-run"),
            cwd=self.root,
            env=self.environment(),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--reference-uci alpha=1", result.stdout)
        self.assertIn("--candidate-uci alpha=2", result.stdout)
        self.assertFalse(self.capture.exists())

    def test_rejects_incomplete_state(self):
        state = json.loads(self.state.read_text())
        state["k"] = 1499
        self.state.write_text(json.dumps(state))

        result = subprocess.run(
            self.command(),
            cwd=self.root,
            env=self.environment(),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("iteration 1500 is required", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_rejects_parameter_name_mismatch(self):
        state = json.loads(self.state.read_text())
        state["names"][1] = "gamma"
        self.state.write_text(json.dumps(state))

        result = subprocess.run(
            self.command(),
            cwd=self.root,
            env=self.environment(),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("names do not match", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_rejects_identical_parameter_sets(self):
        state = json.loads(self.state.read_text())
        state["theta"] = [1.0, 2.0]
        self.state.write_text(json.dumps(state))

        result = subprocess.run(
            self.command(),
            cwd=self.root,
            env=self.environment(),
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("matches the reference defaults", result.stderr)
        self.assertFalse(self.capture.exists())


if __name__ == "__main__":
    unittest.main()
