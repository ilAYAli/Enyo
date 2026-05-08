#!/usr/bin/env python3

import argparse
import re
import subprocess
import sys
from pathlib import Path


def parse_blocks(path: Path) -> list[dict[str, str]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("position "):
            current = {"position": line}
            blocks.append(current)
            continue
        if current is None:
            continue
        if line.startswith("go "):
            current["go"] = line
            continue
        if line.startswith("search_position start: fen="):
            match = re.search(r"fen=([^,]+),", line)
            if match:
                current["fen"] = match.group(1)
            continue
        if line.startswith("bestmove "):
            current["logged_bestmove"] = line.split()[1]
    return [block for block in blocks if "go" in block]


class UciEngine:
    def __init__(self, cmd: str):
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

    def send(self, line: str):
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str) -> list[str]:
        assert self.proc.stdout is not None
        lines: list[str] = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"engine exited while waiting for {prefix}")
            line = line.rstrip("\n")
            lines.append(line)
            if line.startswith(prefix):
                return lines

    def close(self):
        try:
            self.send("quit")
        finally:
            self.proc.terminate()


def bestmove(lines: list[str]) -> str:
    for line in reversed(lines):
        if line.startswith("bestmove "):
            return line.split()[1]
    return "none"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run logged UCI position/go stream in one engine process.")
    parser.add_argument("log", type=Path)
    parser.add_argument("--engine", default="./build/enyo")
    parser.add_argument("--target-fen", default="")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--go-mode", choices=["log", "depth"], default="log")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash", type=int, default=512)
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--searchmoves", default="")
    parser.add_argument("--show-info", action="store_true")
    parser.add_argument("--show-all", action="store_true")
    args = parser.parse_args()

    blocks = parse_blocks(args.log)
    if args.target_fen:
        target_indexes = [i for i, block in enumerate(blocks, 1) if block.get("fen") == args.target_fen]
        if not target_indexes:
            raise SystemExit(f"target FEN not found: {args.target_fen}")
        target_index = target_indexes[0]
    elif args.target_index:
        target_index = args.target_index
    else:
        target_index = len(blocks)

    engine = UciEngine(args.engine)
    try:
        engine.send(f"setoption name Threads value {args.threads}")
        engine.send(f"setoption name Hash value {args.hash}")
        for option in args.option:
            if "=" not in option:
                raise SystemExit(f"--option must be NAME=VALUE: {option}")
            name, value = option.split("=", 1)
            engine.send(f"setoption name {name} value {value}")
        engine.send("isready")
        engine.read_until("readyok")
        engine.send("ucinewgame")
        engine.send("isready")
        engine.read_until("readyok")

        for index, block in enumerate(blocks, 1):
            engine.send(block["position"])
            if args.go_mode == "log":
                go = block["go"]
            else:
                go = f"go depth {args.depth}"
            if index == target_index and args.searchmoves:
                go += " searchmoves " + args.searchmoves
            lines = engine.read_until("bestmove") if False else []
            engine.send(go)
            lines = engine.read_until("bestmove")
            bm = bestmove(lines)
            if args.show_all or index == target_index:
                fen = block.get("fen", "?")
                logged = block.get("logged_bestmove", "?")
                print(f"[{index}/{len(blocks)}] bestmove {bm} logged {logged} fen={fen}")
                if args.show_info or index == target_index:
                    for line in lines:
                        if line.startswith("info depth ") or line.startswith("roottrace "):
                            print(line)
            if index == target_index:
                return 0
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
