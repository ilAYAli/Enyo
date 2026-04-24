#!/usr/bin/env python3

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable


MOVE_LINE_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b")
HASH_RE = re.compile(r"\b(?:hash|Key:)\s*([0-9A-Fa-f]{16})\b")


def find_pgns() -> list[pathlib.Path]:
    root = pathlib.Path("..").resolve()
    return sorted(root.rglob("*.pgn"))


def iter_games(path: pathlib.Path) -> Iterable[list[str]]:
    text = path.read_text(errors="ignore")
    chunks = re.split(r"\n\s*\n(?=\[Event|1\.|$)", text)
    for chunk in chunks:
        moves = MOVE_LINE_RE.findall(chunk)
        if moves:
            yield moves


def run_engine_hash(cmd: list[str], moves: list[str], mode: str) -> str:
    position = "position startpos"
    if moves:
        position += " moves " + " ".join(moves)

    if mode == "stockfish":
        script = "uci\n" + position + "\nd\nquit\n"
    elif mode == "sfhash":
        script = "uci\n" + position + "\nsfhash\nquit\n"
    else:
        script = "uci\n" + position + "\nhash\nquit\n"

    proc = subprocess.run(
        cmd,
        input=script,
        text=True,
        capture_output=True,
        check=False,
    )

    output = proc.stdout + proc.stderr
    matches = HASH_RE.findall(output)
    if not matches:
        raise RuntimeError(f"no hash found for {' '.join(cmd)}\n{output}")
    return matches[-1].upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--enyo", default="./build/enyo")
    parser.add_argument("--stockfish", default="stockfish")
    parser.add_argument("--limit-pgns", type=int, default=0)
    parser.add_argument("--limit-games", type=int, default=0)
    parser.add_argument("--limit-plies", type=int, default=0)
    args = parser.parse_args()

    pgns = find_pgns()
    if args.limit_pgns > 0:
        pgns = pgns[:args.limit_pgns]

    total_games = 0
    skipped_files = 0
    for pgn in pgns:
        game_count = 0
        for moves in iter_games(pgn):
            game_count += 1
            total_games += 1
            if args.limit_games > 0 and total_games > args.limit_games:
                print(f"checked {total_games - 1} games with no mismatches")
                return 0

            upto = len(moves) if args.limit_plies <= 0 else min(len(moves), args.limit_plies)
            for ply in range(upto + 1):
                prefix = moves[:ply]
                enyo_hash = run_engine_hash([args.enyo], prefix, "hash")
                sf_hash = run_engine_hash([args.enyo], prefix, "sfhash")
                if enyo_hash != sf_hash:
                    print(f"mismatch: file={pgn} game={game_count} ply={ply}")
                    print("moves:", " ".join(prefix))
                    print("enyo:", enyo_hash)
                    print("sf:  ", sf_hash)
                    return 1

        if game_count:
            print(f"ok: {pgn} ({game_count} games)")
        else:
            skipped_files += 1

    print(f"checked {total_games} games with no mismatches; skipped {skipped_files} non-UCI PGNs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
