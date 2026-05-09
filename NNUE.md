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

Use Enyo self-play as the first Enyo-owned network path. The self-play games
define the position distribution Enyo actually reaches.

For the current training track, do not directly trust shallow Enyo root scores
as final labels. Use them to collect positions, then relabel sampled positions
with a stronger teacher engine before fine-tuning the Berserk-format net. This
reduces the risk of training Enyo to repeat its own mistakes.

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

## How Training Works

Training is fine-tuning, not building a net from nothing.

1. Start from a known strong Berserk-format `.nn` file.
   - Current source net:
     `/home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn`.
2. Convert self-play games into rows with:
   - Converter script: `tools/selfplay/pgn_to_jsonl.py`.
   - Current source PGN:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.pgn`.
   - Current training rows:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl`.
   - Current packed training rows:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay_packed`.
   - Current conversion stats:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.stats.json`.
   - `fen`: position before the searched move.
   - `score`: Enyo root search score in centipawns from side-to-move view.
   - `wdl`: final game result from side-to-move view.
3. For the teacher-labeled track, sample and relabel positions with:
   - Pipeline script: `tools/nnue2/run_teacher_pipeline.sh`.
   - Sampler: `tools/nnue2/sample_jsonl.py`.
   - Teacher labeler: `tools/nnue2/label_with_uci.py`.
   - Current teacher input:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl`.
   - Current teacher output root:
     `/home/petter/tmp/enyo_teacher/latest`.
   - Sampled rows:
     `/home/petter/tmp/enyo_teacher/latest/sample.jsonl`.
   - Teacher-labeled rows:
     `/home/petter/tmp/enyo_teacher/latest/labeled.jsonl`.
   - Packed teacher-labeled rows:
     `/home/petter/tmp/enyo_teacher/latest/labeled_packed`.
4. Load the `.nn` into the PyTorch model with the same NNUE2 architecture.
   - Training script: `tools/nnue2/train.py`.
5. Train the model to predict teacher centipawn labels.
   - Current run directory:
     `/home/petter/tmp/enyo_teacher/latest`.
   - Candidate net:
     `/home/petter/tmp/enyo_teacher/latest/huber_lr2e-7_e4/model.nn`.
6. Keep a validation slice out of training and compare candidate metrics against
   the original Berserk net.
   - Metric script: `tools/nnue2/eval_dataset.py`.
   - Baseline metrics:
     `/home/petter/tmp/enyo_teacher/latest/baseline_metrics.log`.
   - Candidate metrics:
     `/home/petter/tmp/enyo_teacher/latest/huber_lr2e-7_e4/metrics.log`.
7. Export the trained weights back to Berserk `.nn` format and load it in Enyo.
   - Export is handled by `tools/nnue2/train.py --out-nn`.
8. Accept only after replay gates and games.
   - Replay tool: `/home/petter/code/cpp/chess/replay/build/replay`.
   - Replay logs should be stored under the candidate run directory.

The loss is only a filter. A lower loss means the candidate fits the self-play
labels better, but it does not prove higher Elo. The practical checks are:

- Validation MAE/MSE/MPE25 should improve or at least not regress badly.
- Sign accuracy should not collapse; sign errors often mean wrong side of zero.
- Calibration should stay sane: candidate slope should stay close to the
  reference net, and bucketed MAE/sign should not hide failures near zero.
- Replay gates must not introduce obvious tactical or endgame failures.
- SPRT decides whether the net is actually stronger.

`wdl-lambda` controls how much the target is search score versus game result:

- `1.0`: train only toward search-score labels.
- `0.0`: train only toward game-result labels.
- between them: blend both signals.

Quantization also matters. The PyTorch model trains in float, but the exported
`.nn` stores quantized integer layers plus float head layers. A candidate can
look better before export and still be worse after export, so exported-net eval
and replay are mandatory.

## Pipeline

1. Generate self-play PGNs with `fastchess`.
   - Script: `tools/selfplay/run_selfplay.sh`.
   - Orchestration script: `tools/selfplay/pilot.sh`.
   - Current runner: `/home/petter/source/fastchess/fastchess`.
   - Use the current strongest Enyo binary.
   - Use the current strongest NNUE2 file.
   - Use fixed depth first, usually depth 8-10.
   - Store PGNs under `~/tmp/enyo_selfplay/...`.

2. Convert PGN to JSONL training rows.
   - Script: `tools/selfplay/pgn_to_jsonl.py`.
   - Current source PGN:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.pgn`.
   - Current JSONL rows:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl`.
   - Current stats:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.stats.json`.
   - `fen`: position before the searched move.
   - `score`: root score in centipawns, side-to-move perspective.
   - `wdl`: game result from side-to-move perspective.
   - Keep depth, ply, move, and source metadata for audit.

3. Validate the dataset.
   - Script: `tools/selfplay/audit_jsonl.py`.
   - Main run log:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/pilot.log`.
   - Verify sign convention on sample positions.
   - Check score distribution.
   - Check WDL distribution.
   - Check duplicate rate.
   - Drop mate-score rows and extreme tails for pilots.

4. Relabel a sampled position set with a teacher.
   - Pipeline script: `tools/nnue2/run_teacher_pipeline.sh`.
   - Output root: `/home/petter/tmp/enyo_teacher/latest`.
   - Default teacher: `stockfish`.
   - Default source rows:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl`.
   - Current pilot scale: 150k sampled positions at teacher depth 12.
   - This is the active path until Enyo-owned labels prove better.

5. Train NNUE2.
   - Script: `tools/nnue2/train.py`.
   - Packed dataset script: `tools/nnue2/pack_dataset.py`.
   - Current runner: `tools/nnue2/run_teacher_pipeline.sh`.
   - Current log: `/home/petter/tmp/enyo_teacher/latest/pipeline.log`.
   - Initialize from the current strong Berserk-format net:
     `/home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn`.
   - Current objective: all-layer Huber on teacher centipawn labels.
   - Choose candidates by held-out metrics and replay gates, not loss alone.

6. Export to `.nn`.
   - Export is handled by `tools/nnue2/train.py --out-nn`.
   - Roundtrip script: `tools/nnue2/roundtrip.py`.
   - Dataset metric script: `tools/nnue2/eval_dataset.py`.
   - Current candidate path:
     `/home/petter/tmp/enyo_teacher/latest/huber_lr2e-7_e4/model.nn`.
   - Load via `setoption name nnue2_file value <candidate>.nn`.

7. Gate the candidate.
   - Replay tool: `/home/petter/code/cpp/chess/replay/build/replay`.
   - Replay gate script for first candidate:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/run_replay_gates.sh`.
   - Replay gate log for first candidate:
     `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/replay/gates.log`.
   - Known bad games:
     - `/home/petter/code/cpp/chess/enyo/bugs/Hypersion vs EnyoBot - npmgxvIO.log`
     - `/home/petter/code/cpp/chess/enyo/bugs/EnyoBot vs Lynx_BOT - jjThVRPN.log`
     - `/home/petter/code/cpp/chess/enyo/bugs/JustinBot15 vs EnyoBot - JZaA98Uv.log`
     - `/home/petter/code/cpp/chess/lichess/pgns/EnyoBot vs JustinBot15 - 2WGVezt0.log`
   - Run dataset eval metrics.
   - Run fixed-depth sanity games if useful.
   - Run SPRT against the current reference.

8. Iterate only if the candidate is neutral or positive.
   - Candidate becomes new generator only after it is accepted.
   - Otherwise keep the generator fixed and improve data/training.

## Current Diagnosis

The first 6M-row pilots prove the pipeline, but they are not strong nets yet.
The failure mode is clear:

- MPE25 can improve while centipawn scale and sign accuracy get worse.
- Float-head-only training is too weak; it mostly warps the output scale.
- Lower learning rate all-layer training is safer, but still needs stricter
  exported-net validation before replay or SPRT.

This matches the Stockfish/Berserk lesson: training loss is only a filter.
Stockfish uses very large generated datasets and empirical game testing.
Berserk points at Grapheus/Koivisto-style trainers rather than small one-off
scripts. Enyo's current JSONL path is useful for pilots, but serious iteration
needs packed or streamed data and acceptance stricter than "loss went down".

## Immediate Next Training Plan

The active plan is teacher relabeling, not more direct Enyo-score fine-tuning.
The 6M self-play rows are useful as positions, but shallow Enyo scores can
encode Enyo's current mistakes.

Run:

```sh
tools/nnue2/run_teacher_pipeline.sh
```

Default inputs and outputs:

- Source rows:
  `/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl`.
- Source net:
  `/home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn`.
- Latest output symlink:
  `/home/petter/tmp/enyo_teacher/latest`.
- Candidate net:
  `/home/petter/tmp/enyo_teacher/latest/huber_lr2e-7_e4/model.nn`.
- Pipeline log:
  `/home/petter/tmp/enyo_teacher/latest/pipeline.log`.

The pipeline:

1. Samples 150k unique positions from the 6M-row self-play set.
2. Relabels them with Stockfish at depth 12.
3. Packs the teacher-labeled rows into mmap arrays.
4. Evaluates the original Berserk net on the held-out validation slice.
5. Fine-tunes all layers with a conservative Huber objective.
6. Evaluates the exported candidate on the same validation slice.

Reject candidates that only improve loss. The exported `.nn` must also pass:

- Baseline/candidate metrics with:
   - MAE, MSE, RMSE, MPE25.
   - sign accuracy and wrong-sign count.
   - bias, correlation, and slope.
   - bucketed metrics by absolute target score.
- Replay gates on known bad games.
- SPRT against the current reference only after metrics and replay pass.

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

### Completed: 50k-Game / Roughly 6M-Row Position Set

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

Purpose: produce the position distribution for first Enyo-owned NNUE2
experiments.

Initial result:

- Dataset generation and audit passed: 50,000 games, 6,235,602 rows.
- First all-layer MPE25 candidate improved exported-net MPE25/MSE/MAE on the
  sampled dataset, but sign accuracy dropped from 90.09% to 88.77%.
- Replay gates were mixed: JustinBot blunders improved, but Lynx still produced
  a 1000cp flagged endgame move.
- Status: not accepted as direct-score training data for a final net.

Rejected direct-score variants:

- `all_mpe_lr1e6_lam1_e3`: all-layer MPE25, lower LR, search-score-only target.
- `float_head_mpe_lr1e5_lam09_e5`: float-head-only MPE25, blended target.
- `huber_lr5e7_e3`: metric-positive but failed the Hypersion replay gate.

### Active: Teacher-Labeled Pilot

Running target: `/home/petter/tmp/enyo_teacher/latest`.

Purpose: train on Enyo-reachable positions with stronger teacher scores, then
accept only candidates that beat the source net on metrics and replay gates.

## Acceptance Standard

A network is not accepted because training loss improved. It must satisfy:

- Loads cleanly in Enyo.
- Replays known bug games without new obvious failures.
- Does not regress speed enough to dominate Elo.
- SPRT against the current reference is neutral or positive.

Only accepted networks may become the next self-play generator.
