#!/usr/bin/env python3

import fcntl
import json
import os
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

    def command(self, iterations=1, explicit_parallelism=True):
        command = [
            str(SCRIPT),
            "--params", str(self.params),
            "--state", str(self.state),
            "--fastchess", str(self.fastchess),
            "--iterations", str(iterations),
            "--seed", "1",
        ]
        if explicit_parallelism:
            command.extend(["--rounds", "1", "--concurrency", "1"])
        return command

    def test_writes_state_and_csv_next_to_each_other(self):
        result = subprocess.run(
            self.command(), check=True, capture_output=True, text=True
        )

        state = json.loads(self.state.read_text())
        self.assertEqual(state["k"], 1)
        self.assertEqual(state["names"], ["alpha"])
        self.assertEqual(state["batch"]["iterations"], 1)
        self.assertTrue(state["batch"]["complete"])
        self.assertTrue(self.state.with_suffix(".csv").is_file())
        self.assertIn("final values:", result.stdout)

    def test_iterations_are_added_to_existing_state(self):
        self.state.write_text(json.dumps({
            "k": 7,
            "names": ["alpha"],
            "theta": [1.0],
        }))

        result = subprocess.run(
            self.command(iterations=2), check=True, capture_output=True, text=True
        )

        state = json.loads(self.state.read_text())
        self.assertEqual(state["k"], 9)
        self.assertEqual(state["batch"]["start_k"], 8)
        self.assertEqual(state["batch"]["target_k"], 9)
        self.assertRegex(result.stdout, r"\[1/2; total 8\].*eta=\d+[hms]")

    def test_interrupted_batch_resumes_its_original_target(self):
        self.state.write_text(json.dumps({
            "k": 8,
            "names": ["alpha"],
            "theta": [1.0],
            "batch": {
                "start_k": 8,
                "target_k": 9,
                "iterations": 2,
                "complete": False,
            },
        }))

        subprocess.run(
            self.command(iterations=2), check=True, capture_output=True, text=True
        )

        state = json.loads(self.state.read_text())
        self.assertEqual(state["k"], 9)
        self.assertTrue(state["batch"]["complete"])
        rows = self.state.with_suffix(".csv").read_text().splitlines()
        self.assertEqual(len(rows), 2)

    def test_interrupted_batch_ignores_new_iteration_count(self):
        self.state.write_text(json.dumps({
            "k": 8,
            "names": ["alpha"],
            "theta": [1.0],
            "batch": {
                "start_k": 8,
                "target_k": 9,
                "iterations": 2,
                "complete": False,
            },
        }))

        result = subprocess.run(
            self.command(iterations=17), check=True,
            capture_output=True, text=True,
        )

        state = json.loads(self.state.read_text())
        self.assertEqual(state["k"], 9)
        self.assertTrue(state["batch"]["complete"])
        self.assertIn("1 remaining; --iterations ignored", result.stdout)

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

    def test_infers_parallelism_from_available_processors(self):
        arguments = self.root / "fastchess-arguments"
        executable(
            self.fastchess,
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$FASTCHESS_ARGUMENTS\"\n"
            "echo 'Games: 2, Wins: 1, Losses: 0, Draws: 1'\n",
        )
        environment = os.environ.copy()
        environment["FASTCHESS_ARGUMENTS"] = str(arguments)

        result = subprocess.run(
            self.command(explicit_parallelism=False), check=True,
            capture_output=True, text=True, env=environment,
        )

        args = arguments.read_text().splitlines()
        if hasattr(os, "sched_getaffinity"):
            concurrency = max(1, len(os.sched_getaffinity(0)))
        else:
            concurrency = max(1, os.cpu_count() or 1)
        self.assertEqual(args[args.index("-concurrency") + 1], str(concurrency))
        self.assertEqual(
            args[args.index("-rounds") + 1], str(max(1, concurrency // 2))
        )
        self.assertIn(
            f"settings: concurrency={concurrency} (available processors), "
            f"rounds={max(1, concurrency // 2)} (inferred), "
            f"games/iteration={2 * max(1, concurrency // 2)}, "
            "tc=5+0.05, hash=64",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
