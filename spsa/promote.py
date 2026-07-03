#!/usr/bin/env python3
"""Promote a completed SPSA theta to Enyo's compiled and tuner defaults."""

import argparse
import difflib
import os
import re
import stat
import tempfile
from pathlib import Path

from sprt import load_parameters, load_theta


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parent
DEFAULT_STATE = DIRECTORY / "state.json"
DEFAULT_CONFIG = ROOT / "src/config.hpp"
DEFAULT_PARAMS = DIRECTORY / "params.txt"


def replace_compiled_defaults(
    source: str, parameters: list[tuple[str, int]], theta: list[int]
) -> str:
    updated = source
    for (name, default), tuned in zip(parameters, theta):
        pattern = re.compile(
            rf"^(?P<prefix>\s*int\s+{re.escape(name)}\s*=\s*)"
            rf"(?P<value>-?\d+)(?P<suffix>\s*;[^\n]*)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(updated))
        if len(matches) != 1:
            raise ValueError(
                f"expected one compiled default for {name}, found {len(matches)}"
            )
        current = int(matches[0].group("value"))
        if current != default:
            raise ValueError(
                f"compiled default for {name} is {current}, "
                f"but the parameter definitions say {default}"
            )
        updated = pattern.sub(
            lambda match: (
                f"{match.group('prefix')}{tuned}{match.group('suffix')}"
            ),
            updated,
            count=1,
        )
    return updated


def replace_tuner_defaults(
    source: str, parameters: list[tuple[str, int]], theta: list[int]
) -> str:
    updated = source
    for (name, _default), tuned in zip(parameters, theta):
        pattern = re.compile(
            rf"^(?P<prefix>\s*{re.escape(name)}\s*,)"
            rf"(?P<value>\s*-?\d+(?:\.\d+)?)"
            rf"(?P<suffix>\s*,[^\n]*)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(updated))
        if len(matches) != 1:
            raise ValueError(
                f"expected one tuner default for {name}, found {len(matches)}"
            )
        width = len(matches[0].group("value"))
        replacement = str(tuned).rjust(width)
        updated = pattern.sub(
            lambda match: (
                f"{match.group('prefix')}{replacement}{match.group('suffix')}"
            ),
            updated,
            count=1,
        )
    return updated


def read_text(path: Path, description: str) -> str:
    try:
        return path.read_text()
    except OSError as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error


def atomic_write(path: Path, contents: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", text=True
        )
        with os.fdopen(descriptor, "w") as output:
            os.fchmod(output.fileno(), mode)
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def show_diff(path: Path, before: str, after: str) -> None:
    print("".join(difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=str(path),
        tofile=str(path),
    )), end="")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--require-iteration", type=int, default=1500)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the source changes without writing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_iteration <= 0:
        raise SystemExit("error: --require-iteration must be positive")

    state_path = args.state.expanduser()
    config_path = args.config.expanduser().resolve(strict=False)
    params_path = args.params.expanduser().resolve(strict=False)
    try:
        parameters = load_parameters(params_path)
        names = [name for name, _ in parameters]
        iteration, theta = load_theta(
            state_path, names, args.require_iteration
        )
        config_before = read_text(config_path, "engine config")
        params_before = read_text(params_path, "SPSA parameters")
        config_after = replace_compiled_defaults(
            config_before, parameters, theta
        )
        params_after = replace_tuner_defaults(
            params_before, parameters, theta
        )
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    if args.dry_run:
        show_diff(config_path, config_before, config_after)
        show_diff(params_path, params_before, params_after)
        return 0

    atomic_write(config_path, config_after)
    atomic_write(params_path, params_after)
    print(
        f"Promoted {len(theta)} SPSA values from iteration {iteration} "
        f"to {config_path} and {params_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
