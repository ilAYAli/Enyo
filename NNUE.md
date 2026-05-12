# Enyo NNUE Training

## Goal

Build an Enyo-owned NNUE2 network that is stronger than the current external
starting net when used inside Enyo search.

Static training metrics are only a filter. A net is useful only if it survives
replay gates and then shows Elo in SPRT.

## Runtime Network

The active evaluator is `nnue2_file`, loaded as a Berserk-format `.nn` file.
Enyo's runtime evaluates this net incrementally during alpha-beta search.

Architecture:

- Input features: `16 king buckets * 12 piece/color types * 64 squares = 12288`.
- Feature transformer: sparse active features are summed into an accumulator.
- Hidden width: `1024` values from each side's perspective.
- Search-time input: `us[1024] + them[1024] = 2048`.
- Head: `2048 -> 16 -> 32 -> 1`.
- Output: centipawn evaluation from the side-to-move perspective.

The important point: training does not teach the engine a move directly. It
teaches the network to evaluate positions. Enyo search uses that evaluation to
choose moves.

## How Training Works

Training is supervised learning:

1. Collect positions that Enyo can realistically reach.
2. Ask a stronger teacher, usually Stockfish, to evaluate each position.
3. Convert each position into Enyo NNUE features.
4. Run the current network on those features to get a predicted centipawn score.
5. Compare the prediction with the teacher score.
6. Use backpropagation to make the weights predict the teacher score better.
7. Export the updated weights back to `.nn`.
8. Validate with static metrics, replay gates, and SPRT.

One training row is conceptually:

```json
{
  "fen": "position to train on",
  "score": 42,
  "wdl": 0.5,
  "source_score": 18,
  "teacher": "stockfish",
  "teacher_depth": 16
}
```

Fields:

- `fen`: the chess position.
- `score`: the current training target in centipawns.
- `wdl`: game result target when available; `0.0`, `0.5`, or `1.0`.
- `source_score`: previous score before relabeling, useful for audits.
- `teacher`: source of the current score.
- `teacher_depth`: depth used by the teacher.

For the current depth-16 run, `score` is the Stockfish depth-16 centipawn
evaluation. The network is trained to predict that value.

## Data Sources

Current sources on `pwa-5090`:

- Enyo self-play positions:
  `/home/petter/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl`
- Enyo self-play packed validation:
  `/home/petter/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed`
- Lichess eval positions:
  `/home/petter/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl`
- Stockfish binpack conversion:
  `/home/petter/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/binpack.jsonl`
- Stockfish binpack packed validation:
  `/home/petter/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed`

Raw PGNs are not directly useful for NNUE strength. They first need positions,
teacher labels, filtering, and validation. Lichess logs are useful as replay
gates and hard-position tests, not as bulk training data.

## Current Run

Active run:

```sh
/home/petter/tmp/run_nnue_bucket_d16_20260512.sh
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554
```

Purpose:

1. Sample 1M unique Enyo-reached positions from the 20M self-play pool.
2. Balance the sample by absolute Stockfish depth-12 score:
   - `0-50`: 250k rows
   - `50-100`: 180k rows
   - `100-300`: 300k rows
   - `300-800`: 250k rows
   - `800-1600`: 20k rows
3. Relabel the sample with Stockfish depth 16 using 16 single-threaded shards.
4. Pack the labeled JSONL into mmap arrays for fast training.
5. Train small Huber-loss pilots from the current external starting net.
6. Evaluate metrics and run replay gates.
7. Start SPRT only if the candidate passes the gates.

Why this run exists:

- Previous 30M mixed-data nets were mostly Elo-neutral.
- The likely blocker is target quality and data shape, not trainer mechanics.
- This run spends teacher time on fewer, better-balanced, deeper labels.

## Pipeline Scripts

Training tools live in:

```sh
/home/petter/tmp/enyo-own-net-pipeline/tools/nnue2
```

Important scripts:

| Script | Purpose |
|---|---|
| `sample_jsonl.py` | Randomly sample JSONL rows. |
| `sample_buckets.py` | Sample rows by absolute eval buckets, used by the active depth-16 run. |
| `label_with_uci.py` | Run a UCI teacher engine on FENs and replace `score` with the teacher score. |
| `audit_targets.py` | Summarize score distribution, WDL distribution, source disagreement, and target quality. |
| `pack_dataset.py` | Convert JSONL FEN rows into `.npy` arrays for fast PyTorch training. |
| `train.py` | Load packed data, train the NNUE model, and optionally export a `.nn`. |
| `eval_dataset.py` | Compare a `.nn` against a packed validation set and print MAE, MSE, sign, bias, correlation, and buckets. |
| `mix_jsonl.py` | Mix multiple JSONL sources into one training set. |
| `import_lichess_eval.py` | Convert Lichess eval DB rows into Enyo training JSONL. |
| `binpack_to_jsonl.cpp` | Convert Stockfish binpack data into Enyo JSONL. |
| `export.py` | Export a PyTorch checkpoint to `.nn` when not already exported by `train.py`. |

External validation tools:

| Tool | Purpose |
|---|---|
| `replay` | Re-run known games and identify changed moves, blunders, mistakes, timeouts, and regressions. |
| `sprt` | Run fastchess matches and decide whether a candidate is likely stronger. |
| `fastchess` | Executes the actual engine games used by `sprt`. |

## Main Commands

Sample balanced rows:

```sh
python tools/nnue2/sample_buckets.py \
  --input selfplay_labeled.jsonl \
  --output source_bucketed.jsonl \
  --score-field score \
  --unique-fen \
  --bucket q0050:0:50:250000 \
  --bucket q0100:50:100:180000 \
  --bucket q0300:100:300:300000 \
  --bucket q0800:300:800:250000 \
  --bucket q1600:800:1600:20000
```

Relabel one shard with Stockfish:

```sh
python tools/nnue2/label_with_uci.py \
  --input source_bucketed.jsonl \
  --output shards/label.0.jsonl \
  --engine /usr/games/stockfish \
  --depth 16 \
  --threads 1 \
  --hash 128 \
  --shard-count 16 \
  --shard-index 0 \
  --max-abs-cp 1600
```

Pack labels:

```sh
python tools/nnue2/pack_dataset.py \
  --input labeled.jsonl \
  --out-dir packed
```

Train a pilot:

```sh
python tools/nnue2/train.py \
  --data packed \
  --out candidate.pt \
  --out-nn candidate.nn \
  --init-from-nn current.nn \
  --objective huber \
  --huber-beta 200 \
  --select-metric mae \
  --epochs 4 \
  --lr 1e-7 \
  --batch-size 8192 \
  --target-clamp 1600 \
  --device cuda \
  --val-rows 50000
```

Evaluate a candidate:

```sh
python tools/nnue2/eval_dataset.py \
  --net candidate.nn \
  --data packed_validation \
  --rows 100000 \
  --target-clamp 1600 \
  --buckets
```

Run replay gates:

```sh
replay --force --engine ./enyo_candidate.sh /path/to/game.log
```

Run SPRT:

```sh
sprt \
  --candidate /path/to/enyo \
  --reference /path/to/enyo \
  --candidate-option "nnue2_file=/path/to/candidate.nn" \
  --reference-option "nnue2_file=/path/to/current.nn" \
  --games 2000 \
  --concurrency 6 \
  --threads 4 \
  --tc "10+0.1"
```

## Metrics

Static metrics are useful for rejecting bad nets before spending time on SPRT.

| Metric | Meaning |
|---|---|
| `MAE` | Mean absolute error in centipawns. Lower is better. |
| `MSE` | Mean squared error. Lower is better, but it overemphasizes large misses. |
| `RMSE` | Square root of MSE, also in centipawns. |
| `sign` | Percent of non-zero positions where prediction has the same sign as the target. |
| `bias` | Average prediction minus target. Large bias means the net is systematically optimistic or pessimistic. |
| `corr` | Correlation between prediction and target. Higher is better. |
| `slope` | Calibration slope. A low slope means evals are compressed; high slope means evals are exaggerated. |

Metrics are not enough because chess strength depends on search interaction.
A net can improve MAE and still lose Elo if it damages move ordering, tactical
positions, or calibration in the positions search actually reaches.

## Validation Gates

Every candidate must pass:

1. Load test in Enyo.
2. Static validation against:
   - Enyo self-play validation;
   - Lichess eval validation;
   - binpack validation;
   - the current run's own held-out validation.
3. Replay gates on known bad games.
4. SPRT against the current reference net.

Current replay gate games:

- `EnyoBot vs Lynx_BOT - jjThVRPN.log`
- `Hypersion vs EnyoBot - npmgxvIO.log`
- `JustinBot15 vs EnyoBot - JZaA98Uv.log`
- `EnyoBot vs stage270 - 2DRMYfOm_oot.log`
- `stage270 vs EnyoBot - kp3inZBb.log`

Replay gate outputs are under each candidate directory:

```sh
replay_gates_force/replay.log
replay_gates_force/summary.txt
```

## Previous Results

Recent 30M mixed-data attempts produced safe but near-neutral nets:

- `all_lr2e8_e2`: replay-clean, final SPRT `+2.4 +/- 9.9 Elo`, not enough to
  promote.
- `src_huber_bin35_lr3e8_e2`: replay-clean, stopped at about `-3 Elo` and
  `LLR -0.91`.
- `src_mpe_wdl75_bin35_lr1e6_e2`: rejected before SPRT due to metric
  distortion.

Conclusion: small optimizer changes on the same target mix are not the likely
path to a stronger net. Better teacher targets and better validation are the
current priority.

## Acceptance Rule

A candidate can become the new reference only if:

- replay gates do not add serious new issues;
- static metrics are sane across source-separated validation sets;
- SPRT gives a meaningful positive signal, ideally at least `+5 Elo`;
- search speed is not materially worse.

Neutral nets are archived as experiments, not promoted.

## Next Steps

1. Finish the active depth-16 bucketed run.
2. If a pilot passes gates, let SPRT decide.
3. If the pilot is neutral, scale the same data-quality idea to more depth-16
   or depth-18 labels.
4. Keep collecting Lichess failures for replay gates.
5. Do not spend more time on blind low-LR variants of the old 30M mixed set.
