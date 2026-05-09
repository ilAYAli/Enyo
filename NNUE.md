# Enyo NNUE Plan

## Goal

Build an Enyo-owned network that is stronger than the current shipped network.
The network should improve Enyo's move choices in Enyo's search, not merely
imitate another evaluator on static positions.

## Current Runtime Architecture

Enyo supports two evaluation paths:

- `nnue_file`: the original 512-hidden Enyo NNUE format.
- `nnue2_file`: Berserk-format 1024-hidden network loaded from a `.nn` file.

The active strong path is NNUE2:

- Feature transformer: king-bucketed sparse accumulator.
- Hidden width: 1024 per perspective.
- Concatenated accumulator: `us[1024] + them[1024]`.
- Head: 2048 -> 16 -> 32 -> 1.
- Loader format: Berserk `.nn`.
- Runtime: incremental accumulators, refresh tables, and SIMD paths for ARM
  NEON and x86 AVX2/AVX-512.

This architecture is already strong enough to use externally trained Berserk
networks. The next step is replacing the borrowed weights with Enyo-owned
weights.

## Data Strategy

Use Enyo self-play as the first Enyo-owned network path. Generate games with
the current strongest Enyo, record root search scores, and train on the
positions Enyo actually reaches. This matches the documented Stockfish NNUE
training pattern: self-play generation plus search evaluations, followed by
empirical net testing.

The plan is deliberately empirical. Dataset metrics and loss curves are useful
filters, but a candidate network is accepted only by replay gates and games.

Relevant references:

- Stockfish nnue-pytorch: https://github.com/official-stockfish/nnue-pytorch
- Training datasets: https://github.com/official-stockfish/nnue-pytorch/wiki/Training-datasets
- Basic training procedure: https://github.com/official-stockfish/nnue-pytorch/wiki/Basic-training-procedure-%28train.py%29
- NNUE architecture/training notes: https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md

## Scale Targets

- 1M rows: pipeline validation only.
- 5M-10M rows: first real pilot candidates.
- 50M+ rows: serious candidate scale.
- 100M+ rows: competitive training scale.

Stockfish-scale datasets are far larger; their wiki cites generated datasets in
the billions of positions. Enyo does not need to start there, but small pilots
should not be treated as final evidence.

## Pipeline

1. Generate self-play PGNs with `fastchess`.
   - Use the current strongest Enyo binary.
   - Use the current strongest NNUE2 file.
   - Use fixed depth first, usually depth 8-10.
   - Store PGNs under `~/tmp/enyo_selfplay/...`.

2. Convert PGN to JSONL training rows.
   - `fen`: position before the searched move.
   - `score`: root score in centipawns, side-to-move perspective.
   - `wdl`: game result from side-to-move perspective.
   - Keep depth, ply, move, and source metadata for audit.

3. Validate the dataset.
   - Verify sign convention on sample positions.
   - Check score distribution.
   - Check WDL distribution.
   - Check duplicate rate.
   - Drop mate-score rows and extreme tails for pilots.

4. Train NNUE2.
   - Initialize from the current strong Berserk-format net.
   - Use score/result mixing via `wdl-lambda`; Stockfish's trainer uses the
     same basic idea with a lambda between eval target and game result.
   - Pilot objective can be MPE25 or MSE; choose by held-out metrics and replay
     gates, not by loss alone.
   - Pilot mode may train only the float head, but useful own-net candidates
     require conservative all-layer fine-tuning.

5. Export to `.nn`.
   - Roundtrip-test loader/exporter.
   - Load via `setoption name nnue2_file value nnue/<candidate>.nn`.

6. Gate the candidate.
   - Replay known bad games.
   - Run dataset eval metrics.
   - Run fixed-depth sanity games if useful.
   - Run SPRT against the current reference.

7. Iterate only if the candidate is neutral or positive.
   - Candidate becomes new generator only after it is accepted.
   - Otherwise keep the generator fixed and improve data/training.

## Milestones

### Completed: 1M-Row Pipeline Pilot

Output: `~/tmp/enyo_selfplay/pilot_d8_1m_20260509_141352/selfplay.jsonl`

Result:

- 7,000 games.
- 894,358 JSONL rows.
- Dataset audit passed.
- First training/export/replay loop proved the tooling works.
- Pilot candidates were not accepted; they improved some held-out metrics but
  failed replay gates.

### Active: 50k-Game / Roughly 6M-Row Pilot

Running in tmux `ai:ownnet-6m` on `pwa-5090`.

Output: `~/tmp/enyo_selfplay/d8_6m_20260509_165354`

```sh
tools/selfplay/pilot.sh \
  --runner /home/petter/source/fastchess/fastchess \
  --engine /home/petter/tmp/enyo-own-net-pipeline/build/enyo \
  --nnue2-file /home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --book /home/petter/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd \
  --games 50000 \
  --shard-games 1000 \
  --concurrency 8 \
  --threads 4 \
  --depth 8 \
  --epochs 3 \
  --trainable all \
  --lr 3e-6 \
  --wdl-lambda 0.9 \
  --batch-size 8192 \
  --val-rows 50000 \
  --max-abs-cp 1600 \
  --out-dir /home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354
```

Purpose: produce a first real pilot candidate and validate whether 5M-10M rows
are enough to make replay-safe progress before scaling to 50M+.

## Acceptance Standard

A network is not accepted because training loss improved. It must satisfy:

- Loads cleanly in Enyo.
- Replays known bug games without new obvious failures.
- Does not regress speed enough to dominate Elo.
- SPRT against the current reference is neutral or positive.

Only accepted networks may become the next self-play generator.
