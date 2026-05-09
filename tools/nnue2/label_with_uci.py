"""Relabel JSONL positions with a UCI teacher engine."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


SCORE_RE = re.compile(r"\bscore\s+(cp|mate)\s+(-?\d+)\b")


class UciEngine:
    def __init__(self, path: str, *, threads: int, hash_mb: int) -> None:
        self.proc = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.wait_for("uciok")
        self.setoption("Threads", str(threads))
        self.setoption("Hash", str(hash_mb))
        self.send("isready")
        self.wait_for("readyok")

    def close(self) -> None:
        try:
            self.send("quit")
        finally:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def send(self, command: str) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("engine stdin closed")
        self.proc.stdin.write(command + "\n")
        self.proc.stdin.flush()

    def readline(self) -> str:
        if self.proc.stdout is None:
            raise RuntimeError("engine stdout closed")
        line = self.proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited")
        return line.strip()

    def wait_for(self, token: str) -> None:
        while True:
            if self.readline() == token:
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def label(self, fen: str, *, depth: int) -> tuple[int | None, str | None]:
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")
        score_cp: int | None = None
        mate: str | None = None
        while True:
            line = self.readline()
            match = SCORE_RE.search(line)
            if match:
                kind, value = match.groups()
                if kind == "cp":
                    score_cp = int(value)
                    mate = None
                else:
                    mate = value
            if line.startswith("bestmove "):
                return score_cp, mate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--max-abs-cp", type=int, default=1600)
    ap.add_argument("--progress", type=int, default=1000)
    args = ap.parse_args()

    if not (0 <= args.shard_index < args.shard_count):
        raise ValueError("invalid shard index/count")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input": args.input,
        "output": args.output,
        "engine": args.engine,
        "depth": args.depth,
        "threads": args.threads,
        "hash": args.hash,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "read": 0,
        "selected": 0,
        "written": 0,
        "skipped_mate": 0,
        "skipped_no_score": 0,
        "skipped_cp": 0,
    }

    engine = UciEngine(args.engine, threads=args.threads, hash_mb=args.hash)
    start = time.monotonic()
    try:
        with Path(args.input).open() as src, out.open("w") as dst:
            for idx, line in enumerate(src):
                stats["read"] += 1
                if idx % args.shard_count != args.shard_index:
                    continue
                line = line.strip()
                if not line:
                    continue
                stats["selected"] += 1
                row = json.loads(line)
                score, mate = engine.label(row["fen"], depth=args.depth)
                if mate is not None:
                    stats["skipped_mate"] += 1
                    continue
                if score is None:
                    stats["skipped_no_score"] += 1
                    continue
                if args.max_abs_cp > 0 and abs(score) > args.max_abs_cp:
                    stats["skipped_cp"] += 1
                    continue

                row["source_score"] = row["score"]
                row["score"] = score
                row["teacher"] = Path(args.engine).name
                row["teacher_depth"] = args.depth
                dst.write(json.dumps(row, separators=(",", ":")))
                dst.write("\n")
                stats["written"] += 1

                if args.progress > 0 and stats["selected"] % args.progress == 0:
                    elapsed = max(1e-6, time.monotonic() - start)
                    rate = stats["selected"] / elapsed
                    print(
                        f"shard {args.shard_index}/{args.shard_count}: "
                        f"selected={stats['selected']} written={stats['written']} "
                        f"rate={rate:.1f}/s",
                        flush=True,
                    )
    finally:
        engine.close()

    stats["elapsed_s"] = round(time.monotonic() - start, 3)
    stats_path = out.with_suffix(out.suffix + ".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
