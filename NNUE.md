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

Source-aware 30M run:

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

Results:

- `src_mpe_wdl75_bin35_lr1e6_e2`: rejected before SPRT. It improved MAE, but
  damaged sign accuracy enough that the static metrics were not worth spending
  a full SPRT on.
- `src_huber_bin35_lr3e8_e2`: passed replay and metric gates, then failed to
  show useful Elo in SPRT. Stopped manually at:

```text
[1140/2000] Elo  -3.0 +/- 13.2 | LLR -0.91/2.94 (-31%) | LOS 32.5% | draw 57.2%
```

Takeaway: source-aware controls work technically, but this target mix is not
enough to produce a stronger net. The blocker is now target/data quality, not
the trainer mechanics.

Active depth-16 bucketed teacher run:

```sh
/home/petter/tmp/run_nnue_bucket_d16_20260512.sh
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554
```

Purpose:

- sample 1M unique Enyo-reached positions from the 20M self-play pool;
- balance the sample by absolute Stockfish depth-12 score buckets:
  `0-50`, `50-100`, `100-300`, `300-800`, `800-1600`;
- relabel those positions with Stockfish depth 16 using 16 single-threaded
  shards;
- train small Huber pilots from the current external init net;
- run source metrics and replay gates before any SPRT.

Rationale: the previous 30M mixed target created safe, near-neutral nets. This
run spends teacher time on a smaller but higher-quality and less draw-dominated
target set, so a pilot has a better chance of producing a real Elo signal.

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

1. Let the depth-16 bucketed run finish labeling and pilot training.
2. Start SPRT only if the pilot improves depth-16 validation and does not add
   replay issues.
3. If the pilot is neutral, scale the same idea to more depth-16/depth-18
   labels rather than rerunning optimizer tweaks on the old 30M mix.
4. Keep binpack WDL out of MPE targets until its result convention is proven.
5. Promote only candidates with a realistic chance of at least `+5 Elo`;
   otherwise the SPRT cost is mostly noise measurement.
