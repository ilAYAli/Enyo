#!/usr/bin/env python3
"""SPSA tuner for enyo's UCI search knobs, driven through fastchess.

Each iteration perturbs every parameter by +/-c_k, plays a short match of
the same binary against itself (option-set theta+ vs theta-), and steps
every parameter toward the winning side. Classic SPSA with fishtest-style
per-parameter c_end / r_end schedules.

State is saved to --state after every iteration; rerunning the same
command resumes where it left off. A per-iteration CSV log goes next to
the state file.

Example (smoke test):
  scripts/spsa_tune.py --iterations 4 --rounds 2 --tc 1+0.01 --concurrency 4

Real run (one ~30 thread box, a day or two):
  scripts/spsa_tune.py --iterations 1500 --rounds 16 --tc 5+0.05 --concurrency 30
"""

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ALPHA = 0.602
GAMMA = 0.101

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", default=str(Path.home() / "assets/engines/candidate"))
    ap.add_argument("--book", default=str(Path.home() / "assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd"))
    ap.add_argument("--params", default=str(Path(__file__).parent / "spsa_params.txt"))
    ap.add_argument("--state", default="spsa_state.json")
    ap.add_argument("--fastchess", default="fastchess")
    ap.add_argument("--tc", default="5+0.05")
    ap.add_argument("--hash", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=16,
                    help="rounds per iteration; games = 2x rounds")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=1500)
    ap.add_argument("--match-timeout", type=int, default=3600)
    ap.add_argument("--uci", action="append", default=[],
                    help="extra option for both engines, e.g. --uci nnue_file=/path/net.nn")
    ap.add_argument("--seed", type=int, default=None)
    cfg = ap.parse_args()

    rng = random.Random(cfg.seed)
    params = parse_params(cfg.params)
    state_path = Path(cfg.state)
    log_path = state_path.with_suffix(".csv")

    start_k = 1
    if state_path.exists():
        saved = json.loads(state_path.read_text())
        if saved["names"] != [p["name"] for p in params]:
            sys.exit("state file does not match params file; delete it to restart")
        for p, theta in zip(params, saved["theta"]):
            p["theta"] = theta
        start_k = saved["k"] + 1
        print(f"resuming at iteration {start_k}")

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["iter", "result", "games"]
                                   + [p["name"] for p in params])

    for k in range(start_k, cfg.iterations + 1):
        flips, plus_opts, minus_opts = [], {}, {}
        for p in params:
            _, c_k = schedules(p, k, cfg.iterations)
            flip = rng.choice((-1, 1))
            flips.append(flip)
            lo, hi = p["min"], p["max"]
            plus_opts[p["name"]] = round(min(hi, max(lo, p["theta"] + c_k * flip)))
            minus_opts[p["name"]] = round(min(hi, max(lo, p["theta"] - c_k * flip)))

        result, games = run_match(cfg, plus_opts, minus_opts)

        for p, flip in zip(params, flips):
            a_k, c_k = schedules(p, k, cfg.iterations)
            p["theta"] += (a_k / c_k) * flip * result
            p["theta"] = min(p["max"], max(p["min"], p["theta"]))

        state_path.write_text(json.dumps({
            "k": k,
            "names": [p["name"] for p in params],
            "theta": [p["theta"] for p in params],
        }, indent=1))
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([k, f"{result:+.3f}", games]
                                   + [f"{p['theta']:.2f}" for p in params])

        moved = sorted(params, key=lambda p: abs(p["theta"] - p["default"]) / p["c_end"],
                       reverse=True)[:3]
        drift = ", ".join(f"{p['name']}={p['theta']:.1f}" for p in moved)
        print(f"[{k}/{cfg.iterations}] result={result:+.3f} ({games} games)  {drift}")

    print("\nfinal values:")
    for p in params:
        print(f"  setoption name {p['name']} value {round(p['theta'])}")


if __name__ == "__main__":
    main()
