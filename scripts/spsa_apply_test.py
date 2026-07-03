#!/usr/bin/env python3

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("spsa_apply.py").resolve()


class SpsaApplyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-apply-")
        self.root = Path(self.temporary.name)
        self.state = self.root / "spsa_state.json"
        self.settings = self.root / "settings.json"
        self.params = self.root / "spsa_params.txt"
        self.params.write_text("alpha, 1, 0, 10, 1, 0.03\nbeta, 2, 0, 10, 1, 0.03\n")

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, *extra: str, check: bool = True):
        return subprocess.run(
            [
                str(SCRIPT),
                "--state", str(self.state),
                "--settings", str(self.settings),
                "--params", str(self.params),
                "--require-iteration", "10",
                *extra,
            ],
            check=check,
            capture_output=True,
            text=True,
        )

    def write_state(self, *, iteration=10, names=None, theta=None):
        self.state.write_text(json.dumps({
            "k": iteration,
            "names": names or ["alpha", "beta"],
            "theta": theta or [1.6, 7.4],
        }))

    def test_merges_rounded_theta_and_preserves_other_options(self):
        self.write_state()
        self.settings.write_text(json.dumps({
            "uci_options": {"Threads": 4, "alpha": 1},
        }))

        result = self.run_script()

        self.assertIn("Applied 2 SPSA values from iteration 10", result.stdout)
        self.assertEqual(json.loads(self.settings.read_text()), {
            "uci_options": {"Threads": 4, "alpha": 2, "beta": 7},
        })

    def test_creates_missing_settings_file(self):
        self.write_state(theta=[2.2, 3.8])

        self.run_script()

        self.assertEqual(json.loads(self.settings.read_text()), {
            "uci_options": {"alpha": 2, "beta": 4},
        })

    def test_dry_run_does_not_modify_settings(self):
        self.write_state()
        original = '{"uci_options":{"Hash":128}}\n'
        self.settings.write_text(original)

        result = self.run_script("--dry-run")

        self.assertEqual(self.settings.read_text(), original)
        self.assertEqual(json.loads(result.stdout), {
            "uci_options": {"Hash": 128, "alpha": 2, "beta": 7},
        })

    def test_rejects_incomplete_state(self):
        self.write_state(iteration=9)

        result = self.run_script(check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("iteration 10 is required", result.stderr)
        self.assertFalse(self.settings.exists())

    def test_rejects_state_with_different_parameter_names(self):
        self.write_state(names=["alpha", "gamma"])

        result = self.run_script(check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("names do not match", result.stderr)
        self.assertFalse(self.settings.exists())

    def test_rejects_legacy_settings_schema(self):
        self.write_state()
        original = '{"constants":{"Threads":4}}\n'
        self.settings.write_text(original)

        result = self.run_script(check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("only an object named 'uci_options'", result.stderr)
        self.assertEqual(self.settings.read_text(), original)


if __name__ == "__main__":
    unittest.main()
