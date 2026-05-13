"""Pack self-play JSONL rows into mmap-friendly NNUE2 training arrays."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import enyo_nnue2 as nn2


def count_rows(path: Path, *, skip: int, limit: int) -> int:
    n = 0
    with path.open() as handle:
        for i, line in enumerate(handle):
            if i < skip:
                continue
            if not line.strip():
                continue
            n += 1
            if limit > 0 and n >= limit:
                break
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Source self-play JSONL.")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory containing .npy arrays.")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=32)
    ap.add_argument("--progress", type=int, default=250000)
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = count_rows(src, skip=args.skip, limit=args.limit)
    print(f"packing rows={rows} from {src}", flush=True)
    print(f"out={out}", flush=True)

    shape = (rows, args.max_features)
    w_features = np.lib.format.open_memmap(
        out / "white_features.npy", mode="w+", dtype=np.uint16, shape=shape)
    b_features = np.lib.format.open_memmap(
        out / "black_features.npy", mode="w+", dtype=np.uint16, shape=shape)
    counts = np.lib.format.open_memmap(
        out / "counts.npy", mode="w+", dtype=np.uint8, shape=(rows,))
    stms = np.lib.format.open_memmap(
        out / "stm.npy", mode="w+", dtype=np.uint8, shape=(rows,))
    scores = np.lib.format.open_memmap(
        out / "score.npy", mode="w+", dtype=np.float32, shape=(rows,))
    wdls = np.lib.format.open_memmap(
        out / "wdl.npy", mode="w+", dtype=np.float32, shape=(rows,))
    phase_scales = np.lib.format.open_memmap(
        out / "phase_scale.npy", mode="w+", dtype=np.float32, shape=(rows,))
    source_ids = np.lib.format.open_memmap(
        out / "source_id.npy", mode="w+", dtype=np.uint16, shape=(rows,))

    source_map: dict[str, int] = {}
    max_seen = 0
    written = 0
    with src.open() as handle:
        for i, line in enumerate(handle):
            if i < args.skip:
                continue
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pieces, stm = nn2.parse_fen(row["fen"])
            pieces.sort(key=lambda item: item[2])
            w = nn2.features_from_pieces(pieces, nn2.WHITE)
            b = nn2.features_from_pieces(pieces, nn2.BLACK)
            if len(w) > args.max_features:
                raise ValueError(
                    f"row {i}: {len(w)} features > max {args.max_features}")

            n = len(w)
            max_seen = max(max_seen, n)
            w_features[written, :n] = np.asarray(w, dtype=np.uint16)
            b_features[written, :n] = np.asarray(b, dtype=np.uint16)
            if n < args.max_features:
                w_features[written, n:] = 0
                b_features[written, n:] = 0
            counts[written] = n
            stms[written] = stm
            scores[written] = float(row["score"])
            wdls[written] = float(row.get("wdl", 0.5))
            phase_scales[written] = nn2.phase_scale_from_pieces(pieces)
            source_name = str(
                row.get("source_type") or row.get("teacher") or "unknown")
            if source_name not in source_map:
                source_map[source_name] = len(source_map)
            source_ids[written] = source_map[source_name]

            written += 1
            if args.progress > 0 and written % args.progress == 0:
                print(f"packed {written}/{rows}", flush=True)
            if args.limit > 0 and written >= args.limit:
                break

    meta = {
        "source": str(src),
        "rows": written,
        "skip": args.skip,
        "limit": args.limit,
        "max_features": args.max_features,
        "max_features_seen": max_seen,
        "source_map": source_map,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (out / "source_map.json").write_text(
        json.dumps(source_map, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
