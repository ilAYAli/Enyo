#!/usr/bin/env python3
"""SPSA tuner for enyo's UCI search knobs, driven through fastchess.

Each iteration perturbs every parameter by +/-c_k, plays a short match of
the same binary against itself (option-set theta+ vs theta-), and steps
every parameter toward the winning side. Classic SPSA with fishtest-style
per-parameter c_end / r_end schedules.

Each invocation runs the number of new iterations given by --iterations.
State is saved to --state after every iteration; rerunning an interrupted
command resumes its unfinished batch instead of adding another batch. A
per-iteration CSV log goes next to the state file.

Example (smoke test):
  ./spsa/tune.py --iterations 4 --tc 1+0.01 --concurrency 4

Real run (one ~30 thread box, a day or two):
  ./spsa/tune.py --iterations 1500 --tc 5+0.05
"""

import argparse
import csv
import fcntl
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

ALPHA = 0.602
GAMMA = 0.101
DIRECTORY = Path(__file__).resolve().parent
DEFAULT_PARAMS = DIRECTORY / "params.txt"
DEFAULT_STATE = DIRECTORY / "state.json"

# fastchess summary block: "Results of plus vs minus (...)" followed by
# "Games: 4, Wins: 1, Losses: 1, Draws: 2, ..." (from plus's perspective).
SCORE_RE = re.compile(
    r"Games: (\d+), Wins: (\d+), Losses: (\d+), Draws: (\d+)")


def parse_params(path):
    params = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, default, lo, hi, c_end, r_end = [f.strip() for f in line.split(",")]
        params.append({
            "name": name,
            "default": float(default),
            "theta": float(default),
            "min": float(lo),
            "max": float(hi),
            "c_end": float(c_end),
            "r_end": float(r_end),
        })
    return params


def schedules(param, k, total):
    """Return (a_k, c_k) for iteration k (1-based) of `total`."""
    big_a = 0.1 * total
    c = param["c_end"] * total ** GAMMA
    a = param["r_end"] * param["c_end"] ** 2 * (big_a + total) ** ALPHA
    c_k = c / k ** GAMMA
    a_k = a / (big_a + k) ** ALPHA
    return a_k, c_k


def engine_args(name, engine, options, extra_uci):
    args = [f"cmd={engine}", f"name={name}"]
    for opt_name, value in options.items():
        args.append(f"option.{opt_name}={value}")
    for kv in extra_uci:
        args.append(f"option.{kv}")
    return args


def run_match(cfg, plus_opts, minus_opts):
    cmd = [cfg.fastchess,
           "-engine", *engine_args("plus", cfg.engine, plus_opts, cfg.uci),
           "-engine", *engine_args("minus", cfg.engine, minus_opts, cfg.uci),
           "-each", f"tc={cfg.tc}", "option.Threads=1", f"option.Hash={cfg.hash}",
           "-openings", f"file={cfg.book}", "format=epd", "order=random",
           "-rounds", str(cfg.rounds), "-repeat",
           "-concurrency", str(cfg.concurrency),
           "-recover"]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=cfg.match_timeout)
    scores = SCORE_RE.findall(proc.stdout)
    if not scores:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-2000:])
        raise RuntimeError("could not parse match score from fastchess output")
    games, wins, losses, _draws = map(int, scores[-1])
    return (wins - losses) / max(games, 1), games


def lock_state(state_path: Path):
    """Exclusively lock one tuner state until this process exits."""
    lock_path = state_path.with_suffix(".lock")
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        raise RuntimeError(
            f"another SPSA tuner is already using {state_path}"
        ) from None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def write_state(path: Path, state: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(state, indent=1))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def state_data(k, params, batch):
    return {
        "k": k,
        "names": [p["name"] for p in params],
        "theta": [p["theta"] for p in params],
        "batch": batch,
    }


def available_processors():
    """Return the number of processors available to this process."""
    if hasattr(os, "sched_getaffinity"):
        return max(1, len(os.sched_getaffinity(0)))
    return max(1, os.cpu_count() or 1)


def format_duration(seconds):
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=str(Path.home() / "assets/engines/candidate"))
    ap.add_argument("--book", default=str(Path.home() / "assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd"))
    ap.add_argument("--params", type=Path, default=DEFAULT_PARAMS)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--fastchess", default="fastchess")
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument(
        "--rounds", type=int,
        help="paired rounds per iteration; defaults to concurrency / 2",
    )
    ap.add_argument(
        "--concurrency", type=int,
        help="concurrent games; defaults to available processors",
    )
    ap.add_argument(
        "--iterations", type=int, default=1500,
        help="new iterations to run; ignored when resuming an unfinished batch",
    )
    ap.add_argument("--match-timeout", type=int, default=3600)
    ap.add_argument("--uci", action="append", default=[],
                    help="extra option for both engines, e.g. --uci nnue_file=/path/net.nn")
    ap.add_argument("--seed", type=int, default=None)
    cfg = ap.parse_args()
    if cfg.iterations < 1:
        ap.error("--iterations must be positive")
    concurrency_inferred = cfg.concurrency is None
    if concurrency_inferred:
        cfg.concurrency = available_processors()
    if cfg.concurrency < 1:
        ap.error("--concurrency must be positive")
    rounds_inferred = cfg.rounds is None
    if cfg.rounds is None:
        cfg.rounds = max(1, cfg.concurrency // 2)
    elif cfg.rounds < 1:
        ap.error("--rounds must be positive")

    concurrency_source = " (available processors)" if concurrency_inferred else ""
    rounds_source = " (inferred)" if rounds_inferred else ""
    print(
        f"settings: concurrency={cfg.concurrency}{concurrency_source}, "
        f"rounds={cfg.rounds}{rounds_source}, "
        f"games/iteration={2 * cfg.rounds}, tc={cfg.tc}, hash={cfg.hash}",
        flush=True,
    )

    rng = random.Random(cfg.seed)
    params = parse_params(cfg.params.expanduser())
    state_path = cfg.state.expanduser()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        state_lock = lock_state(state_path)
    except RuntimeError as error:
        raise SystemExit(f"error: {error}") from error
    # Keep the descriptor alive for the entire tuning run. The OS releases the
    # lock automatically after normal completion, Ctrl-C, or process failure.
    log_path = state_path.with_suffix(".csv")

    current_k = 0
    saved = None
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved["names"] != [p["name"] for p in params]:
            sys.exit("state file does not match params file; delete it to restart")
        for p, theta in zip(params, saved["theta"]):
            p["theta"] = theta
        current_k = saved["k"]

    batch = saved.get("batch") if saved else None
    if batch and not batch["complete"]:
        if current_k > batch["target_k"]:
            sys.exit("state file has an invalid unfinished SPSA batch")
        remaining = batch["target_k"] - current_k
        print(
            f"resuming batch at iteration {current_k + 1} "
            f"(target {batch['target_k']}; {remaining} remaining; "
            "--iterations ignored)"
        )
    else:
        batch = {
            "start_k": current_k + 1,
            "target_k": current_k + cfg.iterations,
            "iterations": cfg.iterations,
            "complete": False,
        }
        write_state(state_path, state_data(current_k, params, batch))
        print(
            f"starting {cfg.iterations} new iterations "
            f"({batch['start_k']}-{batch['target_k']})"
        )

    start_k = current_k + 1
    target_k = batch["target_k"]

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "result", "games"]
                                   + [p["name"] for p in params])

    run_started = time.monotonic()
    for k in range(start_k, target_k + 1):
        flips, plus_opts, minus_opts = [], {}, {}
        for p in params:
            _, c_k = schedules(p, k, target_k)
            flip = rng.choice((-1, 1))
            flips.append(flip)
            lo, hi = p["min"], p["max"]
            plus_opts[p["name"]] = round(min(hi, max(lo, p["theta"] + c_k * flip)))
            minus_opts[p["name"]] = round(min(hi, max(lo, p["theta"] - c_k * flip)))

        result, games = run_match(cfg, plus_opts, minus_opts)

        for p, flip in zip(params, flips):
            a_k, c_k = schedules(p, k, target_k)
            p["theta"] += (a_k / c_k) * flip * result
            p["theta"] = min(p["max"], max(p["min"], p["theta"]))

        write_state(state_path, state_data(k, params, batch))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([k, f"{result:+.3f}", games]
                                   + [f"{p['theta']:.2f}" for p in params])

        moved = sorted(params, key=lambda p: abs(p["theta"] - p["default"]) / p["c_end"],
                       reverse=True)[:3]
        drift = ", ".join(f"{p['name']}={p['theta']:.1f}" for p in moved)
        # flush: with stdout redirected to a log file Python block-buffers,
        # and a SIGTERM discards the buffer — the log must stream.
        batch_k = k - batch["start_k"] + 1
        completed = k - start_k + 1
        seconds_per_iteration = (time.monotonic() - run_started) / completed
        eta = format_duration(seconds_per_iteration * (target_k - k))
        print(
            f"[{batch_k}/{batch['iterations']}] "
            f"result={result:+.3f}  {drift}  eta={eta}",
            flush=True,
        )

    batch["complete"] = True
    write_state(state_path, state_data(target_k, params, batch))

    print("\nfinal values:")
    for p in params:
        print(f"  setoption name {p['name']} value {round(p['theta'])}")
    state_lock.close()


if __name__ == "__main__":
    main()
