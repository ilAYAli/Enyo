# Enyo Self-Play Data Pipeline

This is the v1 pipeline for producing Enyo-owned NNUE training rows.

It uses `fastchess` to play Enyo against itself and writes annotated PGNs.
PGN comments such as `{+0.52/8 0.10s}` are root search scores for the
position before the move, from the side-to-move engine's perspective.

## Python Dependency

The converter uses `python-chess`:

```bash
python3 -m venv ~/tmp/selfplay-venv
~/tmp/selfplay-venv/bin/pip install chess
```

## Generate PGN

```bash
tools/selfplay/run_selfplay.sh \
  --runner ~/source/fastchess/fastchess \
  --engine ./build/enyo \
  --nnue2-file nnue/berserk-d43206fe90e4.nn \
  --games 1000 \
  --shard-games 1000 \
  --concurrency 8 \
  --depth 8 \
  --output ~/tmp/enyo_selfplay/selfplay_d8.pgn
```

Use `--tc 10+0.1` instead of `--depth 8` if you want time-control games.
For training data, fixed depth is usually easier to reason about.

The runner auto-detects `fastchess` from `PATH`, `~/source/fastchess/fastchess`,
or the chess workspace. Pass `--runner` if it lives elsewhere.

The runner defaults to `--restart off` so long self-play jobs do not reload the
NNUE file for every game. Use `--shard-games` for large runs; each shard is
written separately and appended to the final PGN.

## Convert PGN To JSONL

```bash
~/tmp/selfplay-venv/bin/python tools/selfplay/pgn_to_jsonl.py \
  ~/tmp/enyo_selfplay/selfplay_d8.pgn \
  --output ~/tmp/enyo_selfplay/selfplay_d8.jsonl \
  --stats ~/tmp/enyo_selfplay/selfplay_d8.stats.json
```

Each JSONL row contains:

- `fen`: position before the move
- `move_uci`: move chosen by search
- `score`: centipawns from side-to-move perspective
- `wdl`: game result from side-to-move perspective (`1.0`, `0.5`, `0.0`)
- `depth`: search depth from the cutechess comment
- `ply`, `fullmove`, `side`, `result`, `termination`, `source`

Default filters:

- skip first 8 plies
- skip mate-score rows
- skip rows with `abs(score) > 10000` in the converter
- the pilot uses `--max-abs-cp 2000` to drop Enyo's saturated `±2045`
  search-score cap

Before training, `pilot.sh` runs `audit_jsonl.py` and fails on invalid FENs,
illegal moves, wrong side-to-move metadata, unexpected depth, too few rows,
large duplicate spikes, or saturated score leakage.
