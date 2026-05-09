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

## What Did Not Work

Do not restart these tracks without a new reason:

- Small cp-only distillation from `default.net`.
- Mixing `default.net` static labels with Stockfish search labels.
- SF-binpack data under the current Python trainer at small scale.
- Ranking child positions as the sole acceptance metric.

Those experiments produced useful tooling, but not a competitive network.

## Competitive Network Path

There are two viable data strategies.

### Option A: Lichess Data

Train on hundreds of millions of Lichess positions.

Pros:

- Broad chess coverage.
- Proven to produce decent small NNUEs.
- Does not require weeks of self-play generation.

Cons:

- Game-result labels are noisy.
- Search-score labels still need an engine pass.
- The result is a broad generic net, not specifically adapted to Enyo.

This is a good scaling path, but not the best first Enyo-owned network.

### Option B: Enyo Self-Play

Generate games with the current strongest Enyo, record root search scores, and
train on those positions.

Pros:

- Labels come from Enyo's own search.
- Data distribution is shaped by positions Enyo actually reaches.
- Iteration is possible: new champion generates the next dataset.
- Avoids mixed-teacher label noise.

Cons:

- Generation is slower.
- Early datasets may be homogeneous unless openings are diverse.
- Requires an end-to-end pipeline and careful validation.

This is the recommended first path.

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
   - Objective: MPE25.
   - Start with `wdl-lambda` around `0.75`.
   - Pilot mode: train only the float head first.
   - Scale mode: train all layers once the pipeline is proven.

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

## First Milestone

Create a 1M-row self-play dataset and train a pilot net:

```sh
tools/selfplay/pilot.sh \
  --runner ~/source/fastchess/fastchess \
  --engine ./build/enyo \
  --nnue2-file nnue/berserk-d43206fe90e4.nn \
  --games 7000 \
  --shard-games 1000 \
  --concurrency 8 \
  --threads 4 \
  --depth 8 \
  --epochs 5 \
  --trainable float-head \
  --out-dir ~/tmp/enyo_selfplay/pilot_d8_1m
```

Expected first result: not necessarily stronger. The useful result is a
validated end-to-end process that can be scaled to 10M, 50M, then 100M+ rows.

## Acceptance Standard

A network is not accepted because training loss improved. It must satisfy:

- Loads cleanly in Enyo.
- Replays known bug games without new obvious failures.
- Does not regress speed enough to dominate Elo.
- SPRT against the current reference is neutral or positive.

Only accepted networks may become the next self-play generator.
