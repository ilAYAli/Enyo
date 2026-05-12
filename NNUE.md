# Enyo NNUE Training

## Goal

Build an Enyo-owned NNUE2 network that is stronger than the current borrowed
Berserk-format network when used inside Enyo search.

Training loss is not enough. A candidate must survive replay gates and then
SPRT before it can become the new reference.

## Runtime Architecture

The active strong evaluator is `nnue2_file`, a Berserk-format `.nn` network:

- Sparse king-bucketed feature transformer.
- Hidden width: 1024 per perspective.
- Concatenated accumulator: `us[1024] + them[1024]`.
- Head: `2048 -> 16 -> 32 -> 1`.
- Runtime: incremental accumulators with NEON and AVX2/AVX-512 SIMD paths.

This can already load strong external Berserk networks. The current training
work is about replacing those weights without losing strength.

## Data Strategy

Use multiple data sources, but validate them separately:

- Enyo self-play positions labeled by Stockfish depth 12.
- Lichess eval positions.
- Stockfish binpack positions.

The current risk is overfitting one source or damaging tactical behavior while
improving aggregate MAE. That is why validation is split by source and replay
gates are mandatory.

## Current Data

On `pwa-5090`:

- Mixed packed training set:
  `/home/petter/tmp/enyo_teacher/mixed_20m_selfplay_5m_lichess_5m_binpack_20260512/packed`
- Self-play validation:
  `/home/petter/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed`
- Lichess eval JSONL:
  `/home/petter/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl`
- Binpack validation:
  `/home/petter/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed`
- Current external init net:
  `/home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn`

## Latest Results

Primary tmux sessions on `pwa-5090`:

```sh
tmux attach -t nnue_test
tmux attach -t nnue_cmd
```

Recent controlled run:

```sh
/home/petter/tmp/enyo_teacher/controlled_30m_20260512_105006
```

Results:

- `output_lr1e5`: replay-clean relative to baseline, but sign/slope
  distortion was too large. Not SPRT tested.
- `float_head_lr1e7`: replay-clean relative to baseline, but SPRT was neutral
  (`~+1 Elo` after `1779/2000` games). Not promoted.

Recent all-layer low-LR run:

```sh
/home/petter/tmp/run_nnue_all_lowlr_20260512.sh
/home/petter/tmp/enyo_teacher/all_lowlr_30m_20260512_155610
```

Candidate:

```sh
all_lr2e8_e2
```

Configuration:

- all weights trainable;
- Huber loss;
- learning rate `2e-8`;
- 2 epochs;
- initialized from the current external net.

SPRT result:

```text
[2000/2000] Elo   2.4 +/-   9.9 | LLR -0.21/2.94 ( -7%) | LOS 68.5% | draw  58.0%
Finished match
Total Time: 03:13:03
```

Takeaway: `all_lr2e8_e2` is replay-clean and should be archived as a candidate,
but it is not strong enough evidence to promote as the new reference. The point
estimate is positive, but the error bar is much larger than the gain and the
LLR is slightly negative versus the SPRT target.

Earlier self-play plus Lichess-only Huber run:

```sh
/home/petter/tmp/enyo_teacher/mixed_20m_selfplay_5m_lichess_standard_20260512/huber_lr1e-7_e5_from_berserk/model.nn
```

SPRT result:

```text
[2000/2000] Elo  -1.39 +/- 9.96, LOS 39.23 %, draw 57.20 %
```

Takeaway: removing binpack and improving static validation loss was still not
enough. The current approach can produce safe near-neutral nets, but not a
meaningfully stronger net.

Target audit finding:

- Packed binpack WDL is strongly skewed (`mean ~= 0.30`, median `0.0`).
- That matters for future MPE/WDL training.
- It does not explain the latest neutral Huber run, because Huber ignores WDL.
- Before using binpack in MPE/WDL training, confirm result POV and either use a
  source-specific `wdl_lambda` or train binpack as CP-only.

Active source-aware run:

```sh
/home/petter/tmp/run_nnue_source_aware_20260512.sh
/home/petter/tmp/enyo_teacher/source_aware_30m_20260512_201907
```

Purpose:

- repack the 30M mix with `source_id.npy` and `source_map.json`;
- train `src_mpe_wdl75_bin35_lr1e6_e2` first;
- if that fails gates, train `src_huber_bin35_lr3e8_e2`;
- compare source-separated metrics against the init net;
- run replay gates before any SPRT;
- start SPRT only if replay has no new issues and metrics remain sane.

This run uses the source-aware trainer commit:

```text
6814334 tools/nnue2: add source-aware training controls
```

## Validation

Every candidate is evaluated against:

- self-play validation MAE/sign/bias/correlation;
- lichess validation MAE/sign/bias/correlation;
- binpack validation MAE/sign/bias/correlation;
- forced replay gates on known bad games.

Replay gate logs are written under each candidate directory:

```sh
replay_gates_force/replay.log
replay_gates_force/summary.txt
```

Current replay gate games:

- `EnyoBot vs Lynx_BOT - jjThVRPN.log`
- `Hypersion vs EnyoBot - npmgxvIO.log`
- `JustinBot15 vs EnyoBot - JZaA98Uv.log`
- `EnyoBot vs stage270 - 2DRMYfOm_oot.log`
- `stage270 vs EnyoBot - kp3inZBb.log`

## Acceptance

A candidate may proceed to SPRT only if:

- it loads cleanly in Enyo;
- source-separated metrics are not clearly worse than baseline;
- replay gates do not introduce new serious blunders;
- search speed is not materially worse.

If SPRT is neutral or positive, the net can become the new generator/reference.
If replay fails, discard the net and improve target/data quality before trying
again.

## Next Steps

1. Keep `all_lr2e8_e2` archived, but do not make it the main reference yet.
2. Stop rerunning small optimizer variations on the same 30M target mix.
3. Improve the pipeline before the next expensive run:
   keep source identity in packed data, support source-specific loss weights,
   and validate each source independently.
4. Improve the targets:
   use deeper or higher-node teacher labels for selected Enyo positions,
   add tactical/endgame coverage, and avoid using biased game-result WDL as a
   global target.
5. Train only small pilots until one improves source-separated validation
   without replay damage.
6. Scale to a large run only after a pilot has a realistic chance of at least
   `+5 Elo`; otherwise the SPRT cost is mostly noise measurement.
