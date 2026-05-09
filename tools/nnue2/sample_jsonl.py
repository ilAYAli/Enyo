"""Deterministically sample self-play JSONL rows for teacher relabeling."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--unique-fen", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir: list[str] = []
    seen: set[str] = set()
    accepted = 0

    with Path(args.input).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if args.unique_fen:
                row = json.loads(line)
                fen = row["fen"]
                if fen in seen:
                    continue
                seen.add(fen)

            accepted += 1
            if len(reservoir) < args.rows:
                reservoir.append(line)
                continue

            j = rng.randrange(accepted)
            if j < args.rows:
                reservoir[j] = line

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for line in reservoir:
            handle.write(line)
            handle.write("\n")

    print(json.dumps({
        "input": args.input,
        "output": args.output,
        "requested_rows": args.rows,
        "written_rows": len(reservoir),
        "accepted_rows": accepted,
        "unique_fen": args.unique_fen,
        "seed": args.seed,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
