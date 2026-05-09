#!/usr/bin/env python3
"""Convert cutechess annotated self-play PGNs into Enyo NNUE JSONL rows."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import chess
    import chess.pgn
except ImportError as exc:  # pragma: no cover - exercised by user environment
    raise SystemExit(
        "python-chess is required: python3 -m pip install chess"
    ) from exc


SCORE_RE = re.compile(
    r"(?P<score>[+-]?(?:M\d+|\d+(?:\.\d+)?))/(?P<depth>\d+)"
    r"(?:\s+(?P<time>\d+(?:\.\d+)?)s)?"
)


def parse_score(comment: str, mate_score_cp: int) -> dict[str, Any] | None:
    match = SCORE_RE.search(comment)
    if not match:
        return None

    raw = match.group("score")
    depth = int(match.group("depth"))
    time_s = float(match.group("time")) if match.group("time") else None

    if "M" in raw:
        sign = -1 if raw.startswith("-") else 1
        mate_ply = int(raw.lstrip("+-")[1:])
        return {
            "score_cp": sign * mate_score_cp,
            "mate_ply": sign * mate_ply,
            "depth": depth,
            "time_s": time_s,
            "raw_score": raw,
        }

    # cutechess writes pawn units (+1.23), while Enyo training rows use cp.
    score_cp = int(round(float(raw) * 100.0))
    return {
        "score_cp": score_cp,
        "mate_ply": None,
        "depth": depth,
        "time_s": time_s,
        "raw_score": raw,
    }


def wdl_for_side(result: str, turn: chess.Color) -> float | None:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if turn == chess.WHITE else 0.0
    if result == "0-1":
        return 1.0 if turn == chess.BLACK else 0.0
    return None


def result_reason(headers: chess.pgn.Headers) -> str:
    return headers.get("Termination") or headers.get("Result", "*")


def iter_rows(
    pgn_path: Path,
    *,
    skip_plies: int,
    min_depth: int,
    max_abs_cp: int,
    include_mates: bool,
    mate_score_cp: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "games": 0,
        "rows": 0,
        "skipped_no_score": 0,
        "skipped_depth": 0,
        "skipped_mate": 0,
        "skipped_cp": 0,
        "skipped_unknown_result": 0,
    }

    with pgn_path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break

            stats["games"] += 1
            result = game.headers.get("Result", "*")
            game_wdl_known = result in {"1-0", "0-1", "1/2-1/2"}
            board = game.board()
            ply = 0
            game_id = (
                game.headers.get("GameId")
                or game.headers.get("Round")
                or str(stats["games"])
            )

            for node in game.mainline():
                move = node.move
                if move is None:
                    continue

                turn = board.turn
                score = parse_score(node.comment, mate_score_cp)
                wdl = wdl_for_side(result, turn)

                if ply >= skip_plies:
                    if score is None:
                        stats["skipped_no_score"] += 1
                    elif score["depth"] < min_depth:
                        stats["skipped_depth"] += 1
                    elif score["mate_ply"] is not None and not include_mates:
                        stats["skipped_mate"] += 1
                    elif abs(score["score_cp"]) > max_abs_cp:
                        stats["skipped_cp"] += 1
                    elif not game_wdl_known or wdl is None:
                        stats["skipped_unknown_result"] += 1
                    else:
                        rows.append(
                            {
                                "fen": board.fen(en_passant="fen"),
                                "move": board.san(move),
                                "move_uci": move.uci(),
                                "score": score["score_cp"],
                                "wdl": wdl,
                                "result": result,
                                "side": "white" if turn == chess.WHITE else "black",
                                "ply": ply,
                                "fullmove": board.fullmove_number,
                                "depth": score["depth"],
                                "time_s": score["time_s"],
                                "mate_ply": score["mate_ply"],
                                "game_id": game_id,
                                "termination": result_reason(game.headers),
                                "source": str(pgn_path),
                            }
                        )
                        stats["rows"] += 1

                board.push(move)
                ply += 1

    return rows, stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract stm-perspective cp/WDL training rows from cutechess PGN comments."
    )
    parser.add_argument("pgn", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--skip-plies", type=int, default=8)
    parser.add_argument("--min-depth", type=int, default=1)
    parser.add_argument("--max-abs-cp", type=int, default=10000)
    parser.add_argument("--include-mates", action="store_true")
    parser.add_argument("--mate-score-cp", type=int, default=10000)
    args = parser.parse_args()

    rows, stats = iter_rows(
        args.pgn,
        skip_plies=args.skip_plies,
        min_depth=args.min_depth,
        max_abs_cp=args.max_abs_cp,
        include_mates=args.include_mates,
        mate_score_cp=args.mate_score_cp,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, separators=(",", ":")) + "\n")

    stats_payload = {"input": str(args.pgn), "output": str(args.output), **stats}
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(
            json.dumps(stats_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(stats_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
