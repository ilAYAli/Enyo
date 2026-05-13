#!/usr/bin/env python3
"""Convert filtered Lichess eval DB rows to Enyo NNUE2 JSONL rows."""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import lzma
import subprocess
import sys
from pathlib import Path
from typing import IO, Any


def open_text(path: Path) -> IO[str]:
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] == ".bz2":
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] in (".xz", ".lzma"):
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if suffixes and suffixes[-1] == ".zst":
        try:
            import zstandard as zstd  # type: ignore
        except ImportError:
            proc = subprocess.Popen(
                ["zstdcat", str(path)],
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if proc.stdout is None:
                raise RuntimeError("zstdcat did not provide stdout")
            return proc.stdout
        dctx = zstd.ZstdDecompressor()
        raw = path.open("rb")
        stream = dctx.stream_reader(raw)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def pick_eval(row: dict[str, Any], *, min_depth: int,
              min_knodes: int) -> dict[str, Any] | None:
    evals = row.get("evals")
    if not isinstance(evals, list):
        return None

    best = None
    best_key = (-1, -1)
    for entry in evals:
        if not isinstance(entry, dict):
            continue
        depth = int(entry.get("depth") or 0)
        knodes = int(entry.get("knodes") or 0)
        if depth < min_depth or knodes < min_knodes:
            continue
        key = (depth, knodes)
        if key > best_key:
            best = entry
            best_key = key
    return best


def score_from_eval(entry: dict[str, Any]) -> int | None:
    pvs = entry.get("pvs")
    if not isinstance(pvs, list) or not pvs:
        return None
    first = pvs[0]
    if not isinstance(first, dict):
        return None
    cp = first.get("cp")
    if cp is None:
        return None
    return int(cp)


def side_to_move_score(fen: str, white_pov_score: int) -> int:
    parts = fen.split()
    if len(parts) < 2:
        raise ValueError(f"invalid FEN: {fen}")
    return white_pov_score if parts[1] == "w" else -white_pov_score


def valid_standard_material(fen: str) -> bool:
    parts = fen.split()
    if len(parts) < 2 or parts[1] not in ("w", "b"):
        return False
    ranks = parts[0].split("/")
    if len(ranks) != 8:
        return False

    white = 0
    black = 0
    white_kings = 0
    black_kings = 0
    for rank in ranks:
        files = 0
        for ch in rank:
            if ch.isdigit():
                files += int(ch)
                continue
            if ch.lower() not in "pnbrqk":
                return False
            files += 1
            if ch.isupper():
                white += 1
                white_kings += ch == "K"
            else:
                black += 1
                black_kings += ch == "k"
        if files != 8:
            return False

    return white <= 16 and black <= 16 and white_kings == 1 and black_kings == 1


def result_wdl(score: int, scale: float) -> float:
    # Lichess eval rows do not carry game result. Use the cp target as the
    # neutral WDL proxy, so WDL mixing does not add unrelated game-result noise.
    import math
    return 1.0 / (1.0 + math.exp(-score / scale))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--min-depth", type=int, default=18)
    ap.add_argument("--min-knodes", type=int, default=0)
    ap.add_argument("--max-abs-cp", type=int, default=1600)
    ap.add_argument("--wdl-scale", type=float, default=400.0)
    ap.add_argument("--unique-fen", action="store_true")
    ap.add_argument("--progress", type=int, default=100000)
    args = ap.parse_args()

    src = Path(args.input).expanduser()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input": str(src),
        "output": str(out),
        "seen": 0,
        "written": 0,
        "skipped_no_eval": 0,
        "skipped_invalid_fen": 0,
        "skipped_no_cp": 0,
        "skipped_extreme": 0,
        "skipped_duplicate": 0,
        "min_depth": args.min_depth,
        "min_knodes": args.min_knodes,
        "max_abs_cp": args.max_abs_cp,
        "unique_fen": args.unique_fen,
    }
    seen_fens: set[str] = set()

    with open_text(src) as handle, out.open("w", encoding="utf-8") as writer:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            stats["seen"] += 1
            row = json.loads(line)
            fen = row.get("fen")
            if not isinstance(fen, str):
                stats["skipped_no_eval"] += 1
                continue
            if not valid_standard_material(fen):
                stats["skipped_invalid_fen"] += 1
                continue
            if args.unique_fen:
                if fen in seen_fens:
                    stats["skipped_duplicate"] += 1
                    continue
                seen_fens.add(fen)

            entry = pick_eval(row, min_depth=args.min_depth,
                              min_knodes=args.min_knodes)
            if entry is None:
                stats["skipped_no_eval"] += 1
                continue

            white_score = score_from_eval(entry)
            if white_score is None:
                stats["skipped_no_cp"] += 1
                continue
            score = side_to_move_score(fen, white_score)
            if abs(score) > args.max_abs_cp:
                stats["skipped_extreme"] += 1
                continue

            out_row = {
                "fen": fen,
                "score": score,
                "wdl": result_wdl(score, args.wdl_scale),
                "white_pov_score": white_score,
                "source": str(src),
                "source_type": "lichess_eval",
                "teacher": "lichess_eval_db",
                "teacher_depth": int(entry.get("depth") or 0),
                "teacher_knodes": int(entry.get("knodes") or 0),
            }
            writer.write(json.dumps(out_row, separators=(",", ":")))
            writer.write("\n")
            stats["written"] += 1

            if args.progress > 0 and stats["written"] % args.progress == 0:
                print(json.dumps(stats), flush=True)
            if args.rows > 0 and stats["written"] >= args.rows:
                break

    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as exc:
        raise SystemExit(f"missing dependency or file: {exc}") from exc
