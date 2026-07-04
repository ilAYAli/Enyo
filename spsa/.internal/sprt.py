#!/usr/bin/env python3
"""Compare a completed SPSA theta with its original defaults through Forge."""

import argparse
import json
import math
import os
import shlex
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parent.parent
DEFAULT_STATE = DIRECTORY / "state.json"
DEFAULT_PARAMS = DIRECTORY / "params.txt"
DEFAULT_ENGINE = "~/assets/engines/candidate"
DEFAULT_NET = "~/code/cpp/chess/enyo/net/berserk-9b84c340af7e.nn"
DEFAULT_BOOK = (
    "~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/"
    "WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd"
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def load_parameters(path: Path) -> list[tuple[str, int]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"could not read SPSA parameters {path}: {error}") from error

    parameters = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"invalid SPSA parameter line: {line}")
        name = fields[0]
        try:
            default = float(fields[1])
        except ValueError as error:
            raise ValueError(f"invalid default for SPSA parameter {name}") from error
        if not name or not math.isfinite(default) or default != round(default):
            raise ValueError(f"invalid default for SPSA parameter {name}")
        parameters.append((name, round(default)))

    names = [name for name, _ in parameters]
    if not names or len(names) != len(set(names)):
        raise ValueError(f"invalid SPSA parameter names in {path}")
    return parameters


def load_theta(
    path: Path, expected_names: list[str], required_iteration: int
) -> tuple[int, list[int]]:
    try:
        state = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"SPSA state does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read SPSA state {path}: {error}") from error

    if not isinstance(state, dict):
        raise ValueError("SPSA state must be a JSON object")
    iteration = state.get("k")
    names = state.get("names")
    theta = state.get("theta")
    if isinstance(iteration, bool) or not isinstance(iteration, int):
        raise ValueError("SPSA state has an invalid iteration 'k'")
    if iteration < required_iteration:
        raise ValueError(
            f"SPSA state is only at iteration {iteration}; "
            f"iteration {required_iteration} is required"
        )
    if names != expected_names:
        raise ValueError("SPSA state names do not match the parameter definitions")
    if not isinstance(theta, list) or len(theta) != len(names):
        raise ValueError("SPSA state theta does not match its parameter names")

    rounded = []
    for name, value in zip(names, theta):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"SPSA value for {name} is not numeric")
        if not math.isfinite(value):
            raise ValueError(f"SPSA value for {name} is not finite")
        rounded.append(round(value))
    return iteration, rounded


def require_local_file(value: str, description: str, *, executable=False) -> None:
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"{description} does not exist: {path}")
    if executable and not os.access(path, os.X_OK):
        raise ValueError(f"{description} is not executable: {path}")


def build_command(args, parameters: list[tuple[str, int]], theta: list[int], iteration: int):
    run = args.run or (
        f"spsa-final-vs-defaults-{args.games}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    command = [args.forge, "sprt", "--run", run]
    if args.wait:
        command.append("--wait")
    command.extend([
        "--comment", f"SPSA iteration {iteration} vs compiled defaults",
        "--book", args.book,
        "--reference", args.engine,
        "--candidate", args.engine,
        "--reference-uci", f"nnue_file={args.net}",
        "--candidate-uci", f"nnue_file={args.net}",
        "--reference-uci", "use_syzygy=false",
        "--candidate-uci", "use_syzygy=false",
    ])
    for (name, default), tuned in zip(parameters, theta):
        command.extend(["--reference-uci", f"{name}={default}"])
        command.extend(["--candidate-uci", f"{name}={tuned}"])
    command.extend([
        "--games", str(args.games),
        "--shards", str(args.shards),
        "--concurrency", str(args.concurrency),
        "--threads", str(args.threads),
        "--hash", str(args.hash),
        "--tc", args.tc,
        "--restart", args.restart,
    ])
    return run, command


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--games", type=positive_int, default=1000)
    parser.add_argument("--run", help="Forge run name; generated by default")
    parser.add_argument("--engine", default=DEFAULT_ENGINE)
    parser.add_argument("--net", default=DEFAULT_NET)
    parser.add_argument("--book", default=DEFAULT_BOOK)
    parser.add_argument("--shards", type=positive_int, default=24)
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--threads", type=positive_int, default=1)
    parser.add_argument("--hash", type=positive_int, default=128)
    parser.add_argument("--tc", default="10+0.1")
    parser.add_argument("--restart", choices=("on", "off"), default="on")
    parser.add_argument("--require-iteration", type=positive_int, default=1500)
    parser.add_argument("--forge", default="forge")
    parser.add_argument(
        "--wait", action=argparse.BooleanOptionalAction, default=False,
        help="wait for Forge completion (default: false)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the Forge command without starting it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    state_path = args.state.expanduser()
    params_path = args.params.expanduser()
    try:
        parameters = load_parameters(params_path)
        names = [name for name, _ in parameters]
        iteration, theta = load_theta(
            state_path, names, args.require_iteration
        )
        defaults = [default for _, default in parameters]
        if theta == defaults:
            raise ValueError(
                "SPSA theta matches the reference defaults; nothing to test"
            )
        require_local_file(args.engine, "engine", executable=True)
        require_local_file(args.net, "NNUE network")
        require_local_file(args.book, "opening book")
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    run, command = build_command(args, parameters, theta, iteration)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    if "/" not in args.forge and shutil.which(args.forge) is None:
        raise SystemExit(f"error: Forge executable not found: {args.forge}")

    print(f"Starting Forge run {run}", flush=True)
    try:
        completed = subprocess.run(command, cwd=Path.home(), check=False)
    except FileNotFoundError as error:
        raise SystemExit(f"error: Forge executable not found: {args.forge}") from error
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
