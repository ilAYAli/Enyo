#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/spsa_promote.py"


class SpsaPromoteTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-promote-")
        self.root = Path(self.temporary.name)
        self.state = self.root / "state.json"
        self.config = self.root / "config.hpp"
        self.params = self.root / "params.txt"
        self.state.write_text(json.dumps({
            "k": 1500,
            "names": ["alpha", "beta"],
            "theta": [3.4, 7.8],
        }))
        self.config.write_text(
            "class Config {\n"
            "    int alpha = 1;\n"
            "    int beta = 2;\n"
            "    int untouched = 9;\n"
            "};\n"
        )
        self.params.write_text(
            "alpha, 1, 0, 10, 1, 0.03\n"
            "beta,  2, 0, 10, 1, 0.03\n"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def command(self, *extra: str):
        return [
            str(SCRIPT),
            "--state", str(self.state),
            "--config", str(self.config),
            "--params", str(self.params),
            *extra,
        ]

    def test_promotes_compiled_and_tuner_defaults(self):
        result = subprocess.run(
            self.command(), check=True, capture_output=True, text=True
        )

        self.assertIn("Promoted 2 SPSA values from iteration 1500", result.stdout)
        self.assertIn("int alpha = 3;", self.config.read_text())
        self.assertIn("int beta = 8;", self.config.read_text())
        self.assertIn("int untouched = 9;", self.config.read_text())
        self.assertIn("alpha, 3,", self.params.read_text())
        self.assertIn("beta,  8,", self.params.read_text())

    def test_dry_run_prints_changes_without_writing(self):
        config_before = self.config.read_text()
        params_before = self.params.read_text()

        result = subprocess.run(
            self.command("--dry-run"),
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("-    int alpha = 1;", result.stdout)
        self.assertIn("+    int alpha = 3;", result.stdout)
        self.assertIn("-alpha, 1,", result.stdout)
        self.assertIn("+alpha, 3,", result.stdout)
        self.assertEqual(self.config.read_text(), config_before)
        self.assertEqual(self.params.read_text(), params_before)

    def test_rejects_default_drift(self):
        self.config.write_text(self.config.read_text().replace(
            "int alpha = 1;", "int alpha = 4;"
        ))

        result = subprocess.run(
            self.command(), check=False, capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compiled default for alpha is 4", result.stderr)

    def test_rejects_incomplete_state(self):
        state = json.loads(self.state.read_text())
        state["k"] = 1499
        self.state.write_text(json.dumps(state))

        result = subprocess.run(
            self.command(), check=False, capture_output=True, text=True
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("iteration 1500 is required", result.stderr)


if __name__ == "__main__":
    unittest.main()
