#!/usr/bin/env python3
"""Deterministically mix multiple JSONL datasets by per-file row quotas.

This script is designed for large training sets. It counts rows first, then
streams an exact-size deterministic sample from each source while interleaving
the selected rows, so it does not hold the mixed dataset in memory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Source:
    path: Path
    requested: int
    total_rows: int
    selected_rows: int
    written: int = 0


def count_rows(path: Path) -> int:
    rows = 0
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return rows


def selected_lines(source: Source, rng: random.Random):
    remaining_available = source.total_rows
    remaining_needed = source.selected_rows
    with source.path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if remaining_needed <= 0:
                break
            if rng.randrange(remaining_available) < remaining_needed:
                yield line
                remaining_needed -= 1
            remaining_available -= 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--source", action="append", required=True,
                    help="Source spec PATH:ROWS. Repeat for each dataset.")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--progress", type=int, default=250000)
    ap.add_argument("--unique-fen", action="store_true",
                    help="Rejected for large streaming mixes; dedupe inputs first.")
    args = ap.parse_args()
    if args.unique_fen:
        raise SystemExit(
            "--unique-fen is not supported by the streaming mixer; "
            "dedupe inputs before mixing")

    rng = random.Random(args.seed)
    stats = {
        "output": args.output,
        "seed": args.seed,
        "streaming": True,
        "sources": [],
    }

    sources: list[Source] = []
    for spec in args.source:
        path_s, rows_s = spec.rsplit(":", 1)
        path = Path(path_s).expanduser()
        requested = int(rows_s)
        total_rows = count_rows(path)
        selected_rows = min(requested, total_rows)
        source = Source(path=path, requested=requested,
                        total_rows=total_rows, selected_rows=selected_rows)
        sources.append(source)
        stats["sources"].append({
            "path": str(path),
            "requested": requested,
            "total_rows": total_rows,
            "selected_rows": selected_rows,
        })

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    select_rngs = [random.Random(rng.randrange(1 << 63)) for _ in sources]
    iterators = [
        selected_lines(source, source_rng)
        for source, source_rng in zip(sources, select_rngs, strict=True)
    ]
    remaining = [source.selected_rows for source in sources]
    total_remaining = sum(remaining)
    written = 0

    with out.open("w", encoding="utf-8") as handle:
        while total_remaining > 0:
            pick = rng.randrange(total_remaining)
            cumulative = 0
            source_idx = 0
            for i, count in enumerate(remaining):
                cumulative += count
                if pick < cumulative:
                    source_idx = i
                    break

            line = next(iterators[source_idx])
            handle.write(line)
            handle.write("\n")
            sources[source_idx].written += 1
            remaining[source_idx] -= 1
            total_remaining -= 1
            written += 1
            if args.progress > 0 and written % args.progress == 0:
                print(f"mixed {written}", flush=True)

    for source, item in zip(sources, stats["sources"], strict=True):
        item["written"] = source.written

    stats["written_rows"] = written
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
