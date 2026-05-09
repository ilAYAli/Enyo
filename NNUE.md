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

## Competitive Network Path

Generate games with the current strongest Enyo, record root search scores, and
train on those positions.

This is the chosen path because:

- labels come from Enyo's own search;
- data distribution is shaped by positions Enyo actually reaches;
- accepted networks can generate the next dataset;
- the result is Enyo-specific instead of a generic Lichess net.

The self-play row does not assert that the played move is objectively best. The
move advances the game and shapes the position distribution. The label is the
root search score for the position before the move, so training teaches the net
to approximate Enyo's searched position value.

Move quality is validated after training with replay, root-child audits, deeper
search, Stockfish checks where useful, and SPRT. A net is only useful if the
learned evaluator improves move choice inside Enyo's search.

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

Create a 1M-row self-play dataset and train a pilot net with the command in
the Scripts section.

Expected first result: not necessarily stronger. The useful result is a
validated end-to-end process that can be scaled to 10M, 50M, then 100M+ rows.

## Acceptance Standard

A network is not accepted because training loss improved. It must satisfy:

- Loads cleanly in Enyo.
- Replays known bug games without new obvious failures.
- Does not regress speed enough to dominate Elo.
- SPRT against the current reference is neutral or positive.

Only accepted networks may become the next self-play generator.

## Scripts

The self-play and NNUE2 training scripts live on the `nnue/own-net-pipeline`
branch until the pipeline is proven:

- `tools/selfplay/pilot.sh`: end-to-end pilot runner.
- `tools/selfplay/run_selfplay.sh`: runs Enyo-vs-Enyo fixed-depth games with
  `fastchess`.
- `tools/selfplay/pgn_to_jsonl.py`: converts annotated PGN to JSONL rows.
- `tools/nnue2/eval_dataset.py`: validates dataset shape and score statistics.
- `tools/nnue2/train.py`: trains an NNUE2 candidate.
- `tools/nnue2/export.py`: exports a Berserk-format `.nn`.
- `tools/nnue2/roundtrip.py`: verifies exported loader compatibility.

Current pwa-5090 pilot command:

```sh
cd ~/tmp/enyo-own-net-pipeline

tools/selfplay/pilot.sh \
  --python ~/.venv/bin/python \
  --runner ~/source/fastchess/fastchess \
  --engine ./build/enyo \
  --nnue2-file ~/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn \
  --games 7000 \
  --shard-games 1000 \
  --concurrency 6 \
  --threads 4 \
  --depth 8 \
  --epochs 5 \
  --trainable float-head \
  --max-rows 0 \
  --val-rows 50000 \
  --batch-size 4096 \
  --device cuda \
  --out-dir ~/tmp/enyo_selfplay/pilot_d8_1m_$(date +%Y%m%d_%H%M%S)
```

Run it in tmux and tee the log:

```sh
tmux new-window -t ai -n ownnet-pilot

cd ~/tmp/enyo-own-net-pipeline
set -o pipefail
tools/selfplay/pilot.sh ... 2>&1 | tee ~/tmp/enyo_ownnet_pilot.log
```
