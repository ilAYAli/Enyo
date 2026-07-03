#!/usr/bin/env python3
"""Apply a completed SPSA state to Enyo's UCI settings."""

import argparse
import json
import math
import os
import stat
import tempfile
from pathlib import Path


DEFAULT_SETTINGS = Path("~/.config/enyo/settings.json")
DEFAULT_PARAMS = Path(__file__).with_name("spsa_params.txt")


def load_json(path: Path, description: str):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"{description} does not exist: {path}") from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read {description} {path}: {error}") from error


def load_parameter_names(path: Path) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"could not read SPSA parameters {path}: {error}") from error

    names = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            names.append(line.split(",", 1)[0].strip())

    if not names or any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"invalid SPSA parameter names in {path}")
    return names


def rounded_theta(state, expected_names: list[str], required_iteration: int) -> tuple[int, dict[str, int]]:
    if not isinstance(state, dict):
        raise ValueError("SPSA state must be a JSON object")

    iteration = state.get("k")
    names = state.get("names")
    theta = state.get("theta")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("SPSA state has an invalid iteration 'k'")
    if iteration < required_iteration:
        raise ValueError(
            f"SPSA state is only at iteration {iteration}; "
            f"iteration {required_iteration} is required"
        )
    if names != expected_names:
        raise ValueError("SPSA state names do not match spsa_params.txt")
    if not isinstance(theta, list) or len(theta) != len(names):
        raise ValueError("SPSA state theta does not match its parameter names")

    values = {}
    for name, value in zip(names, theta):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"SPSA value for {name} is not numeric")
        if not math.isfinite(value):
            raise ValueError(f"SPSA value for {name} is not finite")
        values[name] = round(value)
    return iteration, values


def load_settings(path: Path) -> dict:
    if not path.exists():
        return {"uci_options": {}}

    settings = load_json(path, "settings file")
    if (
        not isinstance(settings, dict)
        or set(settings) != {"uci_options"}
        or not isinstance(settings["uci_options"], dict)
    ):
        raise ValueError(
            "settings file must contain only an object named 'uci_options'"
        )
    return settings


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o600
    temporary = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", text=True
        )
        with os.fdopen(descriptor, "w") as output:
            os.fchmod(output.fileno(), mode)
            json.dump(value, output, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--state", type=Path, default=Path("spsa_state.json"),
        help="SPSA state file (default: ./spsa_state.json)",
    )
    parser.add_argument(
        "--settings", type=Path, default=DEFAULT_SETTINGS,
        help="Enyo settings file (default: ~/.config/enyo/settings.json)",
    )
    parser.add_argument(
        "--params", type=Path, default=DEFAULT_PARAMS,
        help="SPSA parameter definition file",
    )
    parser.add_argument(
        "--require-iteration", type=int, default=1500,
        help="refuse an earlier state (default: 1500)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print the resulting settings without writing them",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.require_iteration < 0:
        raise SystemExit("error: --require-iteration cannot be negative")

    state_path = args.state.expanduser()
    params_path = args.params.expanduser()
    settings_path = args.settings.expanduser().resolve(strict=False)
    try:
        names = load_parameter_names(params_path)
        state = load_json(state_path, "SPSA state")
        iteration, values = rounded_theta(state, names, args.require_iteration)
        settings = load_settings(settings_path)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error

    settings["uci_options"].update(values)
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0

    atomic_write_json(settings_path, settings)
    print(
        f"Applied {len(values)} SPSA values from iteration {iteration} "
        f"to {settings_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
