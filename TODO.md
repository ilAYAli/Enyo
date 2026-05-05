# Enyo roadmap

Tracks the active path ahead. Memory files (`~/.claude/projects/.../memory/`)
have the full history; this file is the summary.

## Goal

A network stronger than `nnue/default.net`. Current `distill_P` beats
our previous champion `A` by +48 Elo pairwise but still trails
`default.net` by an estimated ~280 Elo.

## Track 1 — Engine Elo (proven, incremental)

- [x] Hash fix (`85550c8`) — correctness bug in AccumulatorCache
- [x] TM score-volatility (`33742ca`) — +14.95 Elo at 10+0.1 vs prior
- [ ] TM volatility threshold tuning (60/80/150 cp variants)
- [ ] Forced/low-material time discount
- [ ] LMR base/divisor retuning (magic numbers in `src/search.cpp`)
- [ ] Move-ordering history weights

Each item is a one-variable SPRT at 10+0.1, 1000 games, idle machine.

## Track 2 — Self-play data pipeline (ceiling-breaking path)

Spec: `SELFPLAY_SPEC.md`. Approach: cutechess-cli `-pgnout -eval`,
parse PGN to `(FEN, score, WDL)`. Zero engine changes for v1.

- [x] Parser `tools/selfplay/parse_pgn.py` (`f38f542`, verified on 5 games)
- [ ] 100-game smoke run, inspect filter rates + dedup behavior
- [ ] First 2M-filtered generation against current champion (`distill_P`)
- [ ] Train P's recipe (MPE25 + magnitude + `wdl-lambda=0.5`) on new data
- [ ] SPRT vs `default.net`, 1000 games

Round 1 realistic outcome: tie or slight win. Gains compound over rounds 2+
as champion and data co-evolve.

## Track 3 — Infrastructure / housekeeping

- [x] Cherry-pick hash fix + TM patch to `main`
- [x] Move sources to `src/` (cleanup branch)
- [ ] Merge `cleanup/source-reorg` back to main + rebase feature branches
- [ ] Merge `nnue/foundation` training tooling to main (or prune what
      won't survive the self-play switchover)

## Parked / rejected levers

Don't retry without a new reason — each was an explicit investigation:

- **SF-binpack training data** — label noise scrambles sibling ordering
- **`time/24 + inc` TM base rate** — regressed -59 Elo at n=112
- **ExactBound experiments** — regressed -240 Elo twice
- **NMP R_base / depth gate tuning** — load-bearing
- **NonPV !tthit depth reduction** — load-bearing
- **Child-ordering as SPRT predictor** — it's a floor gate, not a ranker

## Current active branches

- `main` — shipped engine, currently at `33742ca` (TM-volatility)
- `nnue/foundation` — training tooling arc (A..O2..P); parked
- `nnue/selfplay` — SELFPLAY_SPEC.md + PGN parser (Track 2)
- `cleanup/source-reorg` — this branch, src/ move

## Lichess bot

- Running: `33742ca` binary + `distill_P.cal80` net (post phase-2)
