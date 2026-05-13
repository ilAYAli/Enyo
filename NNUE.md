# Enyo NNUE Roadmap

This document tracks the process for building a competitive Enyo-owned NNUE.
It is a roadmap and status file, not the detailed training explainer. The
training mechanics and script details live in the sibling `nnue` repo:
`../nnue/README_nnue_training.html`.

## Goal

Build an Enyo-owned network that is clearly stronger in Enyo search than the
current starting net.

A candidate is not accepted because it has nicer static metrics. It is accepted
only if it passes replay gates and shows useful Elo in SPRT.

Target for promotion:

- replay gates do not introduce serious new issues;
- static metrics are sane on source-separated validation sets;
- SPRT shows a meaningful positive signal, ideally at least `+5 Elo`;
- search speed is not materially worse;
- the candidate is reproducible from documented data and scripts.

## Current Status

The depth-16 bucketed 1M run finished and did not produce a promoted net.

Run artifacts on `pwa-5090`:

```text
run:     /home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554
script:  /home/petter/tmp/run_nnue_bucket_d16_20260512.sh
labels:  988632 Stockfish depth-16 rows
sweep:   /home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/static_sweep_20260513_083803
tmux:    nnue_cmd for active gates/tests
status:  float-head candidates rejected; next test is all_huber_lr1e6_e8
```

Outcome:

- `float_head_huber_lr1e5_e8`: large static metric improvement, rejected before
  SPRT because replay introduced serious tactical issues.
- `float_head_huber_lr3e6_e8`: replay-safe enough, but SPRT trended negative.
  Stopped around:

```text
[1478/2000] Elo  -1.4 +/-  11.2 | LLR -0.88/2.94 (-30%) | LOS 40.3% | draw  59.8%
```

Conclusion: depth-16 relabeling is technically valid, but these float-head
training variants did not create Elo. Do not repeat this exact recipe as another
long run.

## Completed Work

- [x] Stable SPRT wrapper with concise one-line progress.
- [x] Host-aware SPRT defaults for local and `pwa-5090` testing.
- [x] Reference build workflow established for local and `pwa-5090` engines.
- [x] Replay gate workflow established for known bad games and timeout logs.
- [x] `notifai.sh` integrated into long-running remote jobs.
- [x] Self-play generation pipeline created.
- [x] 50k-game self-play batch generated.
- [x] Self-play PGN converted to JSONL rows.
- [x] 20M-row self-play source set labeled with Stockfish depth 12.
- [x] JSONL packing path created for fast PyTorch training.
- [x] Trainer can initialize from an existing `.nn`, train, and export `model.nn`.
- [x] Static evaluation script reports MAE, MSE, RMSE, sign, bias, corr, slope, and bucket metrics.
- [x] Lichess eval data imported for validation.
- [x] Stockfish binpack data converted for validation.
- [x] Mixed 20M/30M experiments completed and judged mostly Elo-neutral.
- [x] Replay-clean neutral candidate tested to `+2.4 +/- 9.9 Elo`; archived, not promoted.
- [x] 1M bucket-balanced depth-16 relabeling completed.
- [x] Aggressive float-head d16 run rejected by replay gates.
- [x] Mild float-head d16 run rejected by SPRT trend.

## Active Goal: Find A Training Signal That Converts To Elo

The current hypothesis is narrower than before: the pipeline works, and static
metrics can improve, but those gains are not reliably becoming stronger moves.
The next useful work is controlled candidate gating, not another blind long run.

Depth-16 bucket run stages:

- [x] Sample 1M bucket-balanced unique rows.
- [x] Relabel those rows with Stockfish depth 16.
- [x] Audit target distribution and source-score disagreement.
- [x] Pack the relabeled data.
- [x] Evaluate the starting net on all validation sets.
- [x] Train and reject `float_head_huber_lr1e5_e8`.
- [x] Train and reject `float_head_huber_lr3e6_e8`.
- [ ] Gate `all_huber_lr1e6_e8`.
- [ ] Start SPRT for `all_huber_lr1e6_e8` only if replay adds no serious issues.
- [ ] If neutral/negative, stop this d16 bucket1m family and change the data/target plan.

Expected outputs:

```text
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/audit_targets.log
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed/
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/baseline/eval/
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/static_sweep_20260513_083803/float_head_huber_lr1e5_e8/model.nn
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/static_sweep_20260513_083803/float_head_huber_lr3e6_e8/model.nn
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/static_sweep_20260513_083803/all_huber_lr1e6_e8/model.nn
```

Do not repeat:

- `float_head_huber_lr1e5_e8`: static metrics looked good, replay got worse.
- `float_head_huber_lr3e6_e8`: replay was acceptable, SPRT was not positive.
- Long training based only on better MAE/sign numbers without replay and SPRT.

## Decision Rules

After each candidate:

1. Compare static metrics against baseline on:
   - current depth-16 bucket validation;
   - self-play depth-12 validation;
   - Lichess eval validation;
   - binpack validation.
2. Reject immediately if metrics are distorted or worse across important sources.
3. Run replay gates on known bad games.
4. Reject if replay adds serious new issues.
5. Start SPRT only if both metrics and replay are acceptable.
6. Promote only if SPRT is meaningfully positive.

Neutral candidates are not promoted. They are kept as experiment artifacts.

## If The D16 Bucket1m Family Is Neutral

Do not keep trying random low-learning-rate variants of the same 1M d16 bucket
set. The next steps should improve training signal quality:

- [ ] Scale the depth-16 bucketed approach beyond 1M rows.
- [ ] Consider depth-18 labels for a smaller high-value subset.
- [ ] Add more high-quality public engine-training positions if storage allows.
- [ ] Use Lichess failures as replay gates and hard validation, not bulk labels.
- [ ] Compare source-separated validation before any SPRT.
- [ ] Keep SPRT as the final arbiter.

## Longer-Term Competitive-Net Plan

### Phase 1: Reliable Pipeline

- [x] Generate Enyo-reached positions.
- [x] Label positions with a stronger teacher.
- [x] Pack training rows efficiently.
- [x] Train from an existing `.nn`.
- [x] Export a runtime `.nn`.
- [x] Validate with replay and SPRT.

### Phase 2: Better Targets

- [ ] Finish the current depth-16 bucketed run.
- [ ] Promote only if SPRT is meaningfully positive.
- [ ] If neutral, scale depth-16/depth-18 labels rather than changing random knobs.

### Phase 3: Data Diversity

- [ ] Add stronger curated public data if it improves validation and SPRT.
- [ ] Keep source-separated validation so one data source cannot hide damage in another.
- [ ] Maintain a hard replay suite from real Enyo losses, timeouts, and blunders.

### Phase 4: Architecture Experiments

Only start architecture changes after the target pipeline can reliably produce
small positive nets. Architecture changes are expensive to validate and can make
old data less useful.

Possible future work:

- [ ] test smaller or different bucket schemes if data starvation appears;
- [ ] test a different head only if export/runtime/search support is clean;
- [ ] keep any architecture change behind SPRT and replay gates.

## Operational Notes

Watch current gates/tests:

```sh
ssh -t petter@pwa-5090 tmux attach -t nnue_cmd
```

Tail the active log:

```sh
ssh petter@pwa-5090 'tail -f /home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/run.log'
```

Check worker utilization:

```sh
ssh petter@pwa-5090 'ps -eo pid,etime,pcpu,pmem,comm,args --sort=-pcpu | head -30'
```

Detailed explanation of how training works:

```text
../nnue/README_nnue_training.html
```
