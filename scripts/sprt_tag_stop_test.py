#!/usr/bin/env python3

import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path


SCRIPT = Path(__file__).with_name("sprt_tag.sh").resolve()


def run(*args: str, cwd: Path, env=None) -> str:
    return subprocess.check_output(args, cwd=cwd, env=env, text=True).strip()


def git(repo: Path, *args: str) -> str:
    return run("git", *args, cwd=repo)


def executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


with tempfile.TemporaryDirectory(prefix="sprt-tag-stop-") as tmp:
    root = Path(tmp)
    repo = root / "repo"
    remote = root / "remote.git"
    home = root / "home"
    engines = home / "assets" / "engines"
    forge_state = root / "forge-state"
    fake_forge = root / "forge"
    notify = root / "notify"
    repo.mkdir()
    engines.mkdir(parents=True)
    forge_state.mkdir()

    run("git", "init", "-b", "main", cwd=repo)
    git(repo, "config", "user.name", "Petter Wahlman")
    git(repo, "config", "user.email", "petter@wahlman.no")

    source = repo / "src" / "engine.cpp"
    source.parent.mkdir()
    source.write_text("0\n")
    git(repo, "add", "src/engine.cpp")
    run("git", "commit", "-m", "base", cwd=repo)
    commits = [git(repo, "rev-parse", "HEAD")]

    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "helper.sh").write_text("tooling only\n")
    git(repo, "add", "scripts/helper.sh")
    run("git", "commit", "-m", "chore: tooling", cwd=repo)
    commits.append(git(repo, "rev-parse", "HEAD"))

    for index, subject in enumerate(("perf: bad", "perf: preserved"), start=1):
        source.write_text(f"{index}\n")
        git(repo, "add", "src/engine.cpp")
        run("git", "commit", "-m", subject, cwd=repo)
        commits.append(git(repo, "rev-parse", "HEAD"))

    run("git", "clone", "--bare", str(repo), str(remote), cwd=root)
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-u", "origin", "main")

    for commit in commits:
        executable(engines / f"enyo_{commit[:7]}", "#!/bin/sh\nexit 0\n")

    executable(notify, "#!/bin/sh\nexit 0\n")
    executable(fake_forge, """#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
state = Path(os.environ["FAKE_FORGE_STATE"])
if sys.argv[1] == "sprt":
    args = sys.argv[2:]
    name = args[args.index("--run") + 1]
    count = state / "count"
    count.write_text(str(int(count.read_text() if count.exists() else "0") + 1))
    run = state / name
    run.mkdir()
    (run / "manifest.json").write_text("{}")
    (run / "status.json").write_text(json.dumps({
        "state": "done",
        "manifest": str(run / "manifest.json"),
        "display": {"fields": ["elo=-30.0", "ci=10.0", "llr=-2.94/2.94"]},
    }))
elif sys.argv[1] == "status":
    print((state / sys.argv[2] / "status.json").read_text())
elif sys.argv[1] == "sh":
    count = state / "move-count"
    count.write_text(str(int(count.read_text() if count.exists() else "0") + 1))
    subprocess.run(["bash", "-c", sys.argv[2]], check=True)
elif sys.argv[1] == "wait":
    pass
else:
    raise SystemExit(2)
""")

    env = os.environ.copy()
    env.update({
        "HOME": str(home),
        "FAKE_FORGE_STATE": str(forge_state),
        "SPRT_TAG_FORGE": str(fake_forge),
        "SPRT_TAG_NOTIFY_COMMAND": str(notify),
        "SPRT_TAG_STATE_ROOT": str(root / "state"),
        "SPRT_TAG_WORK_ROOT": str(root / "work"),
        "SPRT_TAG_TEST_ALL": "0",
    })
    subprocess.run([str(SCRIPT), commits[0]], cwd=repo, env=env, check=True)

    rewritten = git(repo, "rev-list", "--reverse", f"{commits[0]}..HEAD").splitlines()
    assert len(rewritten) == 3
    assert git(repo, "show", "-s", "--format=%s", rewritten[0]) == "chore: tooling"
    assert git(repo, "show", "-s", "--format=%s", rewritten[1]) == "perf: bad elo-30.0, llr-2.94"
    assert git(repo, "show", "-s", "--format=%s", rewritten[2]) == "perf: preserved"
    assert (forge_state / "count").read_text() == "1"
    assert (forge_state / "move-count").read_text() == "2"
    assert not (engines / f"enyo_{commits[2][:7]}").exists()
    assert not (engines / f"enyo_{commits[3][:7]}").exists()
    assert (engines / f"enyo_{rewritten[1][:7]}").exists()
    assert (engines / f"enyo_{rewritten[2][:7]}").exists()
    print("significant-regression stop test passed")
