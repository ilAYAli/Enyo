#!/usr/bin/env python3
"""Run the deterministic search benchmark N times and report average speed.

Usage: bench_avg.py [engine] [runs] [--nodes N] [--perft]
  engine   path to engine binary (default: ./build/enyo)
  runs     number of bench runs  (default: 10)
  --nodes  measure a single-threaded fixed-node *search* (go nodes N)
           instead of the built-in deterministic search suite
  --perft  measure the legacy move-generation benchmark
"""

import re
import statistics
import subprocess
import sys

FEN = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"
SEARCH_BENCH_RE = re.compile(
    r"^Total time \(ms\): (\d+)\n"
    r"Nodes searched: (\d+)\n"
    r"Nodes/second: (\d+)$",
    re.MULTILINE,
)
PERFT_RE = re.compile(r"(\d+) nodes in ([\d.]+) sec, ([\d.]+) Mnps")


def run_bench(engine: str) -> tuple[int, float, float]:
    out = subprocess.run(
        [engine, "bench"], capture_output=True, text=True, check=True
    ).stdout
    m = SEARCH_BENCH_RE.search(out)
    if not m:
        sys.exit(f"error: no bench output from {engine}")
    milliseconds = int(m.group(1))
    nodes = int(m.group(2))
    nodes_per_second = int(m.group(3))
    return nodes, milliseconds / 1000.0, nodes_per_second / 1e6


def run_perft(engine: str) -> tuple[int, float, float]:
    out = subprocess.run(
        [engine, "bench", "perft"], capture_output=True, text=True, check=True
    ).stdout
    m = PERFT_RE.search(out)
    if not m:
        sys.exit(f"error: no perft output from {engine}")
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def run_search(engine: str, nodes: int) -> tuple[int, float, float]:
    cmd_input = (
        "uci\nsetoption name Threads value 1\n"
        f"position fen {FEN}\ngo nodes {nodes}\nquit\n"
    )
    out = subprocess.run(
        [engine], input=cmd_input, capture_output=True, text=True, check=True
    ).stdout
    # use engine-reported numbers so process startup and net loading don't
    # pollute the measurement. info lines carry per-iteration node counts
    # but cumulative time, so sum the former and keep the last of the latter.
    total_nodes = 0
    last_ms = 0
    for line in out.splitlines():
        m = re.search(r"^info depth \d+.* nodes (\d+) .*\btime (\d+)\b", line)
        if m:
            total_nodes += int(m.group(1))
            last_ms = int(m.group(2))
    if not last_ms:
        sys.exit(f"error: no search info output from {engine}")
    sec = last_ms / 1000.0
    return nodes, sec, total_nodes / sec / 1e6


def main() -> None:
    args = sys.argv[1:]
    search_nodes = 0
    perft = False
    if "--nodes" in args:
        idx = args.index("--nodes")
        search_nodes = int(args[idx + 1])
        del args[idx:idx + 2]
    if "--perft" in args:
        args.remove("--perft")
        perft = True
    if search_nodes and perft:
        sys.exit("error: --nodes and --perft are mutually exclusive")
    engine = args[0] if len(args) > 0 else "./build/enyo"
    runs = int(args[1]) if len(args) > 1 else 10

    times: list[float] = []
    mnps: list[float] = []
    nodes = None
    for i in range(1, runs + 1):
        if search_nodes:
            n, sec, m = run_search(engine, search_nodes)
        elif perft:
            n, sec, m = run_perft(engine)
        else:
            n, sec, m = run_bench(engine)
        if nodes is None:
            nodes = n
        elif n != nodes:
            sys.exit(f"error: node count changed between runs ({nodes} vs {n})")
        times.append(sec)
        mnps.append(m)
        print(f"run {i:2d}: {sec:8.6f} sec  {m:10.2f} Mnps")

    print(f"\n{engine}: {nodes} nodes, {runs} runs")
    print(f"  time: avg {statistics.mean(times):.6f}  median {statistics.median(times):.6f}"
          f"  stdev {statistics.stdev(times):.6f}" if runs > 1 else
          f"  time: {times[0]:.6f}")
    print(f"  mnps: avg {statistics.mean(mnps):10.2f}  median {statistics.median(mnps):10.2f}")


if __name__ == "__main__":
    main()
