#!/usr/bin/env python3

import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "train"


def executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TrainWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-workflow-")
        self.base = Path(self.temporary.name)
        self.root = self.base / "enyo"
        self.origin = self.base / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", str(self.origin)],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["git", "init", "-b", "main", str(self.root)],
            check=True, capture_output=True, text=True,
        )
        (self.root / "spsa").mkdir()
        (self.root / "spsa/.internal").mkdir()
        shutil.copy2(SCRIPT, self.root / "spsa/train")
        self.state = self.root / "spsa/state.json"
        self.write_state(2500, 2500)
        executable(
            self.root / "spsa/.internal/tune.py",
            "#!/usr/bin/env python3\n"
            "import json, sys\n"
            "from pathlib import Path\n"
            "state_path = Path(__file__).parents[1] / 'state.json'\n"
            "state = json.loads(state_path.read_text())\n"
            "count = int(sys.argv[sys.argv.index('--iterations') + 1])\n"
            "target = state['k'] + count\n"
            "state['k'] = target\n"
            "state['batch'] = {'start_k': target - count + 1, "
            "'target_k': target, 'iterations': count, 'complete': True}\n"
            "state_path.write_text(json.dumps(state))\n",
        )
        executable(
            self.root / "spsa/.internal/sprt.py",
            "#!/bin/sh\n"
            "printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/sprt-args\"\n",
        )
        self.git("config", "user.name", "Petter Wahlman")
        self.git("config", "user.email", "petter@wahlman.no")
        self.git("add", ".")
        self.git("commit", "-m", "initial")
        self.git("remote", "add", "origin", str(self.origin))
        self.git("push", "-u", "origin", "main")
        self.write_state(2600, 2600)

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True,
            capture_output=True, text=True,
        )

    def write_state(self, iteration: int, target: int) -> None:
        self.state.write_text(json.dumps({
            "k": iteration,
            "batch": {
                "start_k": iteration,
                "target_k": target,
                "iterations": 1,
                "complete": True,
            },
        }))

    def test_checkpoints_tunes_pushes_and_launches_sprt(self):
        result = subprocess.run(
            [str(self.root / "spsa/train")],
            cwd=self.root, check=False, capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Tuning SPSA from iteration 2600 to 3100", result.stdout)
        self.assertIn("Launching detached 1000-game SPRT", result.stdout)
        self.assertEqual(self.git("branch", "--show-current").stdout.strip(), "main")
        self.assertEqual(json.loads(self.state.read_text())["k"], 3100)
        self.assertEqual(
            self.git("status", "--porcelain").stdout.strip(),
            "?? spsa/.internal/sprt-args",
        )
        subjects = self.git("log", "-2", "--format=%s").stdout.splitlines()
        self.assertEqual(subjects, [
            "chore: checkpoint SPSA iteration 3100",
            "chore: checkpoint SPSA iteration 2600",
        ])
        remote_state = subprocess.run(
            ["git", f"--git-dir={self.origin}", "show", "main:spsa/state.json"],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(remote_state.stdout)["k"], 3100)
        sprt_args = (self.root / "spsa/.internal/sprt-args").read_text().splitlines()
        self.assertEqual(sprt_args, ["--run", "spsa-k3100-vs-promoted-defaults"])

    def test_iterations_override(self):
        result = subprocess.run(
            [str(self.root / "spsa/train"), "--iterations", "400"],
            cwd=self.root, check=False, capture_output=True, text=True,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Tuning SPSA from iteration 2600 to 3000", result.stdout)
        self.assertEqual(json.loads(self.state.read_text())["k"], 3000)
        sprt_args = (self.root / "spsa/.internal/sprt-args").read_text().splitlines()
        self.assertEqual(sprt_args, ["--run", "spsa-k3000-vs-promoted-defaults"])


if __name__ == "__main__":
    unittest.main()
