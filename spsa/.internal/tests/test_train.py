#!/usr/bin/env python3

import json
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


SPSA = Path(__file__).resolve().parents[2]


def executable(path: Path, contents: str) -> None:
    path.write_text(contents)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class PublicCommandsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="spsa-commands-")
        self.root = Path(self.temporary.name)
        (self.root / ".internal").mkdir()
        implementations = {
            "train": "tune",
            "sprt": "sprt",
            "promote": "promote",
        }
        for command, implementation in implementations.items():
            shutil.copy2(SPSA / command, self.root / command)
            executable(
                self.root / ".internal" / f"{implementation}.py",
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "from pathlib import Path\n"
                "Path(__file__).with_suffix('.args').write_text(json.dumps(sys.argv[1:]))\n",
            )

    def tearDown(self):
        self.temporary.cleanup()

    def run_command(self, command: str, *arguments: str):
        return subprocess.run(
            [str(self.root / command), *arguments],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )

    def arguments(self, command: str):
        implementation = "tune" if command == "train" else command
        return json.loads(
            (self.root / ".internal" / f"{implementation}.args").read_text()
        )

    def test_train_only_runs_tuner(self):
        result = self.run_command("train", "--iterations", "400")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.arguments("train"), ["--iterations", "400"])
        self.assertFalse((self.root / ".internal/sprt.args").exists())
        self.assertFalse((self.root / ".internal/promote.args").exists())

    def test_sprt_only_runs_validator(self):
        result = self.run_command("sprt", "--dry-run")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.arguments("sprt"), ["--dry-run"])
        self.assertFalse((self.root / ".internal/tune.args").exists())
        self.assertFalse((self.root / ".internal/promote.args").exists())

    def test_promote_only_runs_promoter(self):
        result = self.run_command("promote", "--dry-run")

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.arguments("promote"), ["--dry-run"])
        self.assertFalse((self.root / ".internal/tune.args").exists())
        self.assertFalse((self.root / ".internal/sprt.args").exists())


if __name__ == "__main__":
    unittest.main()
