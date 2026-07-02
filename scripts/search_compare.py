#!/usr/bin/env python3
"""A/B-compare two engine binaries: search identity + speed.

Usage: search_compare.py engineA engineB [--depth D] [--runs N] [--nodes N] [--net PATH]

Identity: runs a fixed-depth single-threaded search on a position suite and
diffs the full info output (nodes, scores, PVs, bestmove). Any divergence
means the binaries do not search the same tree.

Speed: interleaved fixed-node searches, engine-reported time, medians.

--net: optional path to an .nn file. Tested on the production NNUE path
when set; defaults to whatever each engine loads on startup.
"""

import re
import statistics
import subprocess
import sys

POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 9",
    "8/2k5/2p5/2Kp4/3P4/8/5P2/8 w - - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N w - - 0 1",
    "r2q1rk1/ppp2ppp/3bbn2/3p4/8/1B1P1N2/PPP2PPP/R1BQR1K1 w - - 5 12",
    "8/8/8/8/8/6k1/4q3/6K1 b - - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
]

STRIP_RE = re.compile(r" (nps|time) \d+")
INFO_RE = re.compile(r"^info depth \d+.* nodes (\d+) .*\btime (\d+)\b")


NET_OPT = ""


def run(engine: str, fen: str, go: str) -> str:
    pre = "uci\nsetoption name Threads value 1\n"
    if NET_OPT:
        pre += f"setoption name nnue_file value {NET_OPT}\n"
    cmd = pre + f"isready\nposition fen {fen}\n{go}\nquit\n"
    return subprocess.run(
        [engine], input=cmd, capture_output=True, text=True, check=True
    ).stdout


def identity(a: str, b: str, depth: int) -> bool:
    ok = True
    for fen in POSITIONS:
        outs = []
        for engine in (a, b):
            raw = run(engine, fen, f"go depth {depth}")
            lines = [STRIP_RE.sub("", ln) for ln in raw.splitlines()
                     if ln.startswith(("info depth", "bestmove"))]
            outs.append("\n".join(lines))
        if outs[0] == outs[1]:
            print(f"  identical: {fen}")
        else:
            ok = False
            print(f"  DIFFERS:   {fen}")
            for la, lb in zip(outs[0].splitlines(), outs[1].splitlines()):
                if la != lb:
                    print(f"    A: {la}\n    B: {lb}")
                    break
    return ok


def search_time(engine: str, fen: str, nodes: int) -> float:
    out = run(engine, fen, f"go nodes {nodes}")
    last_ms = 0
    for line in out.splitlines():
        m = INFO_RE.search(line)
        if m:
            last_ms = int(m.group(2))
    if not last_ms:
        sys.exit(f"error: no search output from {engine}")
    return last_ms / 1000.0


def speed(a: str, b: str, runs: int, nodes: int) -> None:
    fen = POSITIONS[1]
    times: dict[str, list[float]] = {a: [], b: []}
    wins = 0
    for i in range(1, runs + 1):
        ta = search_time(a, fen, nodes)
        tb = search_time(b, fen, nodes)
        wins += tb < ta
        times[a].append(ta)
        times[b].append(tb)
        print(f"  run {i:2d}: A {ta:8.3f}s   B {tb:8.3f}s")
    ma, mb = statistics.median(times[a]), statistics.median(times[b])
    print(f"\n  A ({a}): median {ma:.3f}s")
    print(f"  B ({b}): median {mb:.3f}s")
    print(f"  B faster in {wins}/{runs} runs, median delta {100 * (ma - mb) / ma:+.2f}%")


def engine_id(engine: str) -> str:
    out = subprocess.run(
        [engine], input="uci\nquit\n", capture_output=True, text=True, check=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("id name"):
            return line.removeprefix("id name ").strip()
    return "unknown"


def main() -> None:
    global NET_OPT
    args = sys.argv[1:]
    if "--net" in args:
        idx = args.index("--net")
        NET_OPT = args[idx + 1]
        del args[idx:idx + 2]
    opts = {"--depth": 14, "--runs": 8, "--nodes": 3_000_000}
    for flag in list(opts):
        if flag in args:
            idx = args.index(flag)
            opts[flag] = int(args[idx + 1])
            del args[idx:idx + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    a, b = args
    if NET_OPT:
        print(f"net override: {NET_OPT}\n")

    ida, idb = engine_id(a), engine_id(b)
    print(f"A: {a}\n   {ida}")
    print(f"B: {b}\n   {idb}")
    if ida == idb:
        sys.exit("\nWARNING: both binaries report the same version/commit "
                 "(stale build?). Rebuild B, then rerun.")
    print()

    print(f"identity (go depth {opts['--depth']}, Threads=1):")
    same = identity(a, b, opts["--depth"])
    print(f"\nspeed (go nodes {opts['--nodes']}, {opts['--runs']} interleaved runs):")
    speed(a, b, opts["--runs"], opts["--nodes"])
    print(f"\nverdict: {'TREE-IDENTICAL' if same else 'TREES DIFFER — not a pure speed change'}")


if __name__ == "__main__":
    main()
