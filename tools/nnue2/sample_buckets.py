#!/usr/bin/env python3
"""Sample JSONL rows by absolute score buckets."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Bucket:
    name: str
    lo: float
    hi: float
    rows: int
    seen: int = 0
    accepted: int = 0
    reservoir: list[str] = field(default_factory=list)

    def contains(self, value: float) -> bool:
        return self.lo <= value < self.hi


def parse_hi(value: str) -> float:
    if value in ("inf", "+inf"):
        return float("inf")
    return float(value)


def parse_bucket(spec: str) -> Bucket:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"bucket must be NAME:LO:HI:ROWS, got {spec}")
    name, lo, hi, rows = parts
    return Bucket(name=name, lo=float(lo), hi=parse_hi(hi), rows=int(rows))


def maybe_take(bucket: Bucket, line: str, rng: random.Random) -> None:
    bucket.accepted += 1
    if len(bucket.reservoir) < bucket.rows:
        bucket.reservoir.append(line)
        return
    idx = rng.randrange(bucket.accepted)
    if idx < bucket.rows:
        bucket.reservoir[idx] = line


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bucket", action="append", required=True,
                    help="Bucket spec NAME:LO:HI:ROWS using abs(score).")
    ap.add_argument("--score-field", default="score")
    ap.add_argument("--unique-fen", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--progress", type=int, default=1000000)
    args = ap.parse_args()

    buckets = [parse_bucket(spec) for spec in args.bucket]
    rng = random.Random(args.seed)
    seen_fens: set[str] = set()
    total_seen = 0
    total_used = 0
    skipped_duplicate = 0
    skipped_missing_score = 0
    skipped_no_bucket = 0

    with Path(args.input).open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total_seen += 1
            row = json.loads(line)

            if args.unique_fen:
                fen = row.get("fen")
                if not isinstance(fen, str):
                    continue
                if fen in seen_fens:
                    skipped_duplicate += 1
                    continue
                seen_fens.add(fen)

            if args.score_field not in row:
                skipped_missing_score += 1
                continue
            abs_score = abs(float(row[args.score_field]))

            matched = False
            for bucket in buckets:
                if bucket.contains(abs_score):
                    bucket.seen += 1
                    maybe_take(bucket, line, rng)
                    total_used += 1
                    matched = True
                    break
            if not matched:
                skipped_no_bucket += 1

            if args.progress > 0 and total_seen % args.progress == 0:
                print(
                    f"seen={total_seen} used={total_used} "
                    + " ".join(
                        f"{b.name}:{len(b.reservoir)}/{b.rows}"
                        for b in buckets),
                    flush=True)

    selected: list[str] = []
    for bucket in buckets:
        selected.extend(bucket.reservoir)
    rng.shuffle(selected)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for line in selected:
            handle.write(line)
            handle.write("\n")

    stats = {
        "input": args.input,
        "output": args.output,
        "score_field": args.score_field,
        "seed": args.seed,
        "unique_fen": args.unique_fen,
        "seen": total_seen,
        "written": len(selected),
        "skipped_duplicate": skipped_duplicate,
        "skipped_missing_score": skipped_missing_score,
        "skipped_no_bucket": skipped_no_bucket,
        "buckets": [
            {
                "name": bucket.name,
                "lo": bucket.lo,
                "hi": bucket.hi,
                "requested": bucket.rows,
                "seen": bucket.seen,
                "written": len(bucket.reservoir),
            }
            for bucket in buckets
        ],
    }
    out.with_suffix(out.suffix + ".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
