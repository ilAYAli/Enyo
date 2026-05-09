#!/usr/bin/env python3
"""Audit Enyo self-play JSONL before NNUE training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import chess
except ImportError as exc:  # pragma: no cover - exercised by user environment
    raise SystemExit(
        "python-chess is required: python3 -m pip install chess"
    ) from exc


REQUIRED_FIELDS = {
    "fen",
    "move_uci",
    "score",
    "wdl",
    "side",
    "depth",
    "ply",
    "fullmove",
}


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    idx = min(len(values) - 1, max(0, int(q * len(values))))
    return sorted(values)[idx]


def load_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid json: {exc}")
                continue

            missing = REQUIRED_FIELDS.difference(row)
            if missing:
                errors.append(
                    f"line {line_no}: missing fields: {sorted(missing)}")
                continue

            rows.append(row)

    return rows, errors


def validate_rows(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, 1):
        fen = row["fen"]
        try:
            board = chess.Board(fen)
        except ValueError as exc:
            errors.append(f"row {idx}: invalid fen {fen!r}: {exc}")
            continue

        expected_side = "white" if board.turn == chess.WHITE else "black"
        if row["side"] != expected_side:
            errors.append(
                f"row {idx}: side {row['side']!r} != FEN side "
                f"{expected_side!r}")

        try:
            move = chess.Move.from_uci(row["move_uci"])
        except ValueError as exc:
            errors.append(
                f"row {idx}: invalid move_uci {row['move_uci']!r}: {exc}")
            continue

        if move not in board.legal_moves:
            errors.append(
                f"row {idx}: illegal move {row['move_uci']} in {fen}")

        score = row["score"]
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            errors.append(f"row {idx}: non-finite score {score!r}")

        wdl = row["wdl"]
        if wdl not in (0.0, 0.5, 1.0):
            errors.append(f"row {idx}: invalid wdl {wdl!r}")

    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--min-rows", type=int, default=1)
    ap.add_argument("--expected-depth", type=int, default=0)
    ap.add_argument("--max-duplicate-pct", type=float, default=20.0)
    ap.add_argument("--cap-cp", type=int, default=2045)
    ap.add_argument("--max-capped-pct", type=float, default=0.5)
    args = ap.parse_args()

    rows, errors = load_rows(args.jsonl)
    errors.extend(validate_rows(rows))

    n = len(rows)
    if n < args.min_rows:
        errors.append(f"rows {n} < required minimum {args.min_rows}")

    depths = Counter(int(row["depth"]) for row in rows)
    if args.expected_depth and set(depths) != {args.expected_depth}:
        errors.append(
            f"depth histogram {dict(depths)} != expected "
            f"{{{args.expected_depth}: {n}}}")

    fens = [row["fen"] for row in rows]
    duplicate_pct = pct(n - len(set(fens)), n)
    if duplicate_pct > args.max_duplicate_pct:
        errors.append(
            f"duplicate FEN rate {duplicate_pct:.2f}% > "
            f"{args.max_duplicate_pct:.2f}%")

    scores = [int(row["score"]) for row in rows]
    abs_scores = [abs(score) for score in scores]
    capped_pct = pct(sum(abs(score) >= args.cap_cp for score in scores), n)
    if capped_pct > args.max_capped_pct:
        errors.append(
            f"capped score rate {capped_pct:.2f}% > "
            f"{args.max_capped_pct:.2f}% at cap {args.cap_cp}")

    by_wdl: dict[float, list[int]] = {0.0: [], 0.5: [], 1.0: []}
    for row in rows:
        by_wdl[float(row["wdl"])].append(int(row["score"]))

    if all(len(v) >= 100 for v in by_wdl.values()):
        loss_mean = statistics.mean(by_wdl[0.0])
        draw_mean = statistics.mean(by_wdl[0.5])
        win_mean = statistics.mean(by_wdl[1.0])
        if not (loss_mean < draw_mean < win_mean):
            errors.append(
                "WDL/score ordering inverted: "
                f"loss={loss_mean:.1f} draw={draw_mean:.1f} "
                f"win={win_mean:.1f}")

    side_counts = Counter(str(row["side"]) for row in rows)
    wdl_counts = Counter(float(row["wdl"]) for row in rows)
    print(f"audit rows={n}")
    print(f"audit depth={dict(sorted(depths.items()))}")
    print(f"audit side={dict(side_counts)}")
    print(f"audit wdl={dict(sorted(wdl_counts.items()))}")
    if scores:
        print(
            "audit score "
            f"min={min(scores)} max={max(scores)} "
            f"mean={statistics.mean(scores):.2f} "
            f"abs_p50={quantile(abs_scores, 0.50)} "
            f"abs_p90={quantile(abs_scores, 0.90)} "
            f"abs_p99={quantile(abs_scores, 0.99)}")
    print(f"audit duplicate_fen={duplicate_pct:.2f}%")
    print(f"audit capped_scores={capped_pct:.2f}% cap={args.cap_cp}")
    if all(len(v) >= 100 for v in by_wdl.values()):
        print(
            "audit wdl_score_mean "
            f"loss={statistics.mean(by_wdl[0.0]):.2f} "
            f"draw={statistics.mean(by_wdl[0.5]):.2f} "
            f"win={statistics.mean(by_wdl[1.0]):.2f}")

    if errors:
        for error in errors[:50]:
            print(f"audit error: {error}", file=sys.stderr)
        if len(errors) > 50:
            print(f"audit error: ... {len(errors) - 50} more", file=sys.stderr)
        return 1

    print("audit ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
