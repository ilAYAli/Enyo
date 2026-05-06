#!/usr/bin/env python3

'''
python3 tools/root_child_audit.py \
    --fen '1r6/2R2p1k/p2p1P2/2pb2BP/3b2P1/3N1P2/6K1/8 w - - 1 32' \
    --played d3f4
'''
import argparse
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass

import chess


MATE_CP = 30000


@dataclass
class Score:
    cp: int
    raw: str


@dataclass
class ChildScore:
    score: Score
    pv: list[str]


class UciEngine:
    def __init__(self, cmd: str):
        self.cmd = cmd
        self.proc = subprocess.Popen(
            [cmd],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self.read_until("uciok")

    def close(self):
        try:
            self.send("quit")
        finally:
            self.proc.terminate()

    def send(self, line: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        assert self.proc.stdout is not None
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"{self.cmd} exited while waiting for {prefix}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def setoption(self, name: str, value: str):
        self.send(f"setoption name {name} value {value}")

    def isready(self):
        self.send("isready")
        self.read_until("readyok")

    def newgame(self):
        self.send("ucinewgame")
        self.isready()

    def go_depth(self, fen: str, depth: int, searchmoves: list[str] | None = None) -> list[str]:
        self.send(f"position fen {fen}")
        suffix = ""
        if searchmoves:
            suffix = " searchmoves " + " ".join(searchmoves)
        self.send(f"go depth {depth}{suffix}")
        return self.read_until("bestmove")


def parse_score(line: str) -> Score | None:
    m = re.search(r"score (cp|mate) (-?\d+)", line)
    if not m:
        return None
    kind, value_s = m.groups()
    value = int(value_s)
    if kind == "cp":
        return Score(value, f"{value:+d}")
    cp = MATE_CP - min(abs(value), 1000)
    if value < 0:
        cp = -cp
    return Score(cp, f"mate {value:+d}")


def parse_bestmove(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if line.startswith("bestmove "):
            parts = line.split()
            return parts[1] if len(parts) >= 2 else None
    return None


def parse_root_depths(lines: list[str]) -> list[tuple[int, Score, str]]:
    result = []
    for line in lines:
        if not line.startswith("info depth ") or " multipv " in line:
            continue
        score = parse_score(line)
        if score is None:
            continue
        parts = line.split()
        if "depth" not in parts or "pv" not in parts:
            continue
        depth = int(parts[parts.index("depth") + 1])
        pv = parts[parts.index("pv") + 1:]
        if pv:
            result.append((depth, score, pv[0]))
    return result


def parse_stockfish_multipv(lines: list[str], depth: int) -> dict[str, Score]:
    scores: dict[str, Score] = {}
    for line in lines:
        if not line.startswith(f"info depth {depth} "):
            continue
        if " multipv " not in line or " pv " not in line:
            continue
        score = parse_score(line)
        if score is None:
            continue
        parts = line.split()
        pv = parts[parts.index("pv") + 1:]
        if pv:
            scores[pv[0]] = score
    return scores


def parse_depth_score_pv(lines: list[str], depth: int) -> tuple[Score | None, list[str]]:
    score = None
    pv = []
    for line in lines:
        if line.startswith(f"info depth {depth} "):
            maybe_score = parse_score(line)
            if maybe_score is None:
                continue
            score = maybe_score
            parts = line.split()
            if "pv" in parts:
                pv = parts[parts.index("pv") + 1:]
    if score is not None:
        return score, pv
    for line in reversed(lines):
        if not line.startswith("info depth "):
            continue
        maybe_score = parse_score(line)
        if maybe_score is None:
            continue
        parts = line.split()
        pv = parts[parts.index("pv") + 1:] if "pv" in parts else []
        return maybe_score, pv
    return None, []


def score_child(engine: UciEngine, board: chess.Board, move: chess.Move, depth: int) -> ChildScore:
    engine.newgame()
    child = board.copy(stack=False)
    child.push(move)
    lines = engine.go_depth(child.fen(), depth)
    score, pv = parse_depth_score_pv(lines, depth)
    if score is None:
        raise RuntimeError(f"no score for child {move.uci()}")
    return ChildScore(Score(-score.cp, score.raw), pv)


def score_forced_root(engine: UciEngine, fen: str, move: chess.Move, depth: int) -> ChildScore:
    engine.newgame()
    lines = engine.go_depth(fen, depth, [move.uci()])
    score, pv = parse_depth_score_pv(lines, depth)
    if score is None:
        raise RuntimeError(f"no forced-root score for {move.uci()}")
    return ChildScore(score, pv)


def fmt_cp(value: int | None) -> str:
    if value is None:
        return "   ?"
    return f"{value:+5d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit root child move ordering against Stockfish.")
    parser.add_argument("--fen", required=True)
    parser.add_argument("--played", default="")
    parser.add_argument("--enyo", default="./build/enyo")
    parser.add_argument("--stockfish", default=shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    parser.add_argument("--enyo-depth", type=int, default=12)
    parser.add_argument("--sf-depth", type=int, default=18)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash", type=int, default=512)
    parser.add_argument(
        "--enyo-option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="extra UCI option sent to Enyo; may be repeated",
    )
    parser.add_argument("--limit", type=int, default=0, help="limit table rows after sorting by SF score")
    parser.add_argument("--show-pv", action="store_true", help="show child PVs for SF best, Enyo root best, and played move")
    parser.add_argument(
        "--forced-root",
        action="store_true",
        help="also score every legal move with Enyo `go depth N searchmoves MOVE`",
    )
    args = parser.parse_args()

    board = chess.Board(args.fen)
    legal = list(board.legal_moves)
    legal_uci = {move.uci() for move in legal}
    if args.played and args.played not in legal_uci:
        raise SystemExit(f"played move is not legal in FEN: {args.played}")

    enyo = UciEngine(args.enyo)
    sf = UciEngine(args.stockfish)
    try:
        for engine in (enyo, sf):
            engine.setoption("Threads", str(args.threads))
            engine.setoption("Hash", str(args.hash))
            engine.isready()
        for option in args.enyo_option:
            if "=" not in option:
                raise SystemExit(f"--enyo-option must be NAME=VALUE: {option}")
            name, value = option.split("=", 1)
            enyo.setoption(name, value)
        enyo.isready()

        enyo.newgame()
        enyo_root_lines = enyo.go_depth(args.fen, args.enyo_depth)
        enyo_best = parse_bestmove(enyo_root_lines)
        enyo_depths = parse_root_depths(enyo_root_lines)

        sf.newgame()
        sf.setoption("MultiPV", str(len(legal)))
        sf.isready()
        sf_root_lines = sf.go_depth(args.fen, args.sf_depth)
        sf_scores = parse_stockfish_multipv(sf_root_lines, args.sf_depth)
        if len(sf_scores) < len(legal):
            print(f"warning: Stockfish returned {len(sf_scores)}/{len(legal)} root moves at depth {args.sf_depth}", file=sys.stderr)

        enyo_child_scores = {}
        enyo_forced_scores = {}
        for move in legal:
            enyo_child_scores[move.uci()] = score_child(enyo, board, move, args.enyo_depth)
            if args.forced_root:
                enyo_forced_scores[move.uci()] = score_forced_root(enyo, args.fen, move, args.enyo_depth)

        sf_best_cp = max((score.cp for score in sf_scores.values()), default=None)
        rows = []
        for move in legal:
            uci = move.uci()
            sf_cp = sf_scores.get(uci).cp if uci in sf_scores else None
            loss = None if sf_best_cp is None or sf_cp is None else sf_best_cp - sf_cp
            rows.append((sf_cp if sf_cp is not None else -math.inf, move, loss))
        rows.sort(reverse=True, key=lambda row: row[0])
        if args.limit:
            rows = rows[:args.limit]

        print(f"FEN: {args.fen}")
        print(f"side: {'white' if board.turn == chess.WHITE else 'black'}")
        if args.played:
            print(f"played: {args.played} ({board.san(chess.Move.from_uci(args.played))})")
        print(f"enyo root best: {enyo_best}")
        print()
        print("Enyo root by depth:")
        for depth, score, move in enyo_depths:
            marker = " PLAYED" if args.played and move == args.played else ""
            print(f"  d{depth:2d} {score.raw:>8} {move}{marker}")
        print()
        forced_header = f" {'enyo_forced':>12s}" if args.forced_root else ""
        print(f"{'move':6s} {'san':10s} {'sf_cp':>7s} {'loss':>6s} {'enyo_child':>11s}{forced_header} flags")
        print("-" * (71 if args.forced_root else 58))
        for _, move, loss in rows:
            uci = move.uci()
            sf_cp = sf_scores.get(uci).cp if uci in sf_scores else None
            enyo_cp = enyo_child_scores[uci].score.cp
            forced_cp = enyo_forced_scores[uci].score.cp if args.forced_root else None
            flags = []
            if uci == args.played:
                flags.append("PLAYED")
            if uci == enyo_best:
                flags.append("ENYO_ROOT")
            if loss == 0:
                flags.append("SF_BEST")
            if loss is not None and loss >= args.threshold:
                flags.append(f"BLUNDER>{args.threshold}")
            forced_part = f" {fmt_cp(forced_cp):>12s}" if args.forced_root else ""
            print(
                f"{uci:6s} {board.san(move):10s} {fmt_cp(sf_cp):>7s}"
                f" {str(loss) if loss is not None else '?':>6s}"
                f" {fmt_cp(enyo_cp):>11s}{forced_part} {' '.join(flags)}"
            )
        if args.show_pv:
            interesting = set()
            if args.played:
                interesting.add(args.played)
            if enyo_best:
                interesting.add(enyo_best)
            for _, move, loss in rows:
                if loss == 0:
                    interesting.add(move.uci())
                    break
            print()
            print("Child PVs from Enyo searches:")
            for uci in sorted(interesting):
                move = chess.Move.from_uci(uci)
                child = enyo_child_scores[uci]
                pv = " ".join(child.pv[:20])
                print(f"  {uci:6s} {board.san(move):10s} {child.score.raw:>8} pv {pv}")
                if args.forced_root:
                    forced = enyo_forced_scores[uci]
                    forced_pv = " ".join(forced.pv[:20])
                    print(f"  {uci:6s} {board.san(move):10s} forced {forced.score.raw:>8} pv {forced_pv}")
    finally:
        enyo.close()
        sf.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
