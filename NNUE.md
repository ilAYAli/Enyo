# Enyo NNUE Roadmap

This document tracks the process for building a competitive Enyo-owned NNUE.
It is a roadmap and status file, not the detailed training explainer. The
training mechanics and script details live in the sibling `nnue` repo:
`../nnue/README_nnue_training.html`.

## Goal

Build an Enyo-owned network that is clearly stronger in Enyo search than the
current starting net.

A candidate is not accepted because it has nicer static metrics. It is accepted
only if it shows useful Elo in SPRT. Replay gates are diagnostic while `replay`
is being changed, and must not block or promote NNUE candidates until replay is
stable again.

Target for promotion:

- static metrics are sane on source-separated validation sets;
- SPRT shows a meaningful positive signal, ideally at least `+5 Elo`;
- search speed is not materially worse;
- the candidate is reproducible from documented data and scripts.

## Current Status

The first source-mix smoke test found a small positive signal, but repeated
larger tests have not proven a keeper yet. The latest cp800/source-mix candidate
looked good in a 1000-game screen, then failed to reproduce in a clean 4000-game
run, so it is archived as noisy/inconclusive.

Run artifacts on `pwa-5090`:

```text
run:    /home/petter/tmp/enyo_teacher/objective_sweep_cp800_5m_20260514_080450
script: /home/petter/code/cpp/chess/nnue/tools/nnue2/run_cp800_5m_sweep_pwa.sh
net:    /home/petter/tmp/enyo_teacher/objective_sweep_cp800_5m_20260514_080450/mix1m_huber_cp800_lr1e6_e8/model.nn
tmux:   nnue_test
status: noisy/inconclusive; not promoted
```

Outcome:

- `source_mix_1m/binpack1m_all_huber_lr1e6_e8` finished:

```text
[2000/2000] Elo   4.3 +/-  10.0 | LLR  0.27/2.94 (  9%) | LOS 80.2% | draw  56.8%
```

- `objective_cp800_huber_vs_refnet_e5_screen` finished:

```text
[1000/1000] Elo  13.9 +/-  13.9 | LLR  0.79/2.94 ( 27%) | LOS 97.5% | draw  58.2%
```

- The clean follow-up `objective_cp800_huber_vs_refnet_e5_official4k` was stopped
  after it turned negative:

```text
[953/4000] about -5.5 Elo, LOS about 23%, draw about 55%
```

Conclusion: the current cp800/source-mix family is not a keeper. Do not repeat
it without changing the data source or objective.

Important data finding:

- The imported Stockfish binpack contains many rows where `score=0` but
  Stockfish depth-10 evaluates the same positions around `+/-600` to `+/-1000`
  cp. Training on those zero labels is harmful.
- `tools/nnue2/binpack_to_jsonl.cpp` now supports `--min-abs-cp`; the active
  run filters out small/zero binpack scores and uses binpack only as tactical
  nonzero data.

Active run:

```text
run:    /home/petter/tmp/enyo_teacher/nonzero_binpack_sweep_20260514_164049
script: /home/petter/code/cpp/chess/nnue/tools/nnue2/run_nonzero_binpack_sweep_pwa.sh
tmux:   nnue_test
plan:   d16 + lichess eval + filtered nonzero binpack, no replay gates, static gate, then 4000-game SPRT if static passes
```

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
- [x] Source-mix smoke test completed.
- [x] Binpack-only 1M candidate finished `+4.3 +/- 10.0 Elo`; archived as promising, not promoted.
- [x] D16+binpack 1M candidate rejected by replay gate.
- [x] cp800/source-mix 5M candidate screened positive but failed clean reproduction; archived, not promoted.
- [x] Binpack zero-score label problem identified.
- [x] Binpack converter gained `--min-abs-cp` filtering.

## Active Goal: Find A Training Signal That Converts To Elo

The current hypothesis is narrower than before: the pipeline works, and static
metrics can improve, but those gains are not reliably becoming stronger moves.
The next useful work is controlled candidate gating, not another blind long run.

Current run:

- [x] Stop treating imported binpack zero scores as valid centipawn labels.
- [ ] Convert filtered binpack rows with `--min-abs-cp 50`.
- [ ] Mix `900k` d16 rows, `2.5M` Lichess eval rows, and `1.6M` filtered binpack rows.
- [ ] Train conservative Huber/MPE candidates from the starting net.
- [ ] Reject candidates with source-separated static metric distortion.
- [ ] Run one `4000` game SPRT only if static metrics pass.
- [ ] Promote only if SPRT is meaningfully positive.

Depth-16 bucket run stages:

- [x] Sample 1M bucket-balanced unique rows.
- [x] Relabel those rows with Stockfish depth 16.
- [x] Audit target distribution and source-score disagreement.
- [x] Pack the relabeled data.
- [x] Evaluate the starting net on all validation sets.
- [x] Train and reject `float_head_huber_lr1e5_e8`.
- [x] Train and reject `float_head_huber_lr3e6_e8`.
- [x] Gate `all_huber_lr1e6_e8`.
- [x] Reject `all_huber_lr1e6_e8` by SPRT trend.
- [x] Stop this d16 bucket1m family and change the data/target plan.

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
- `all_huber_lr1e6_e8`: replay-safe but SPRT was negative/neutral.
- `mix_d16_500k_binpack_500k_all_huber_lr1e6_e8`: replay added a serious tactical regression.
- `objective_cp800_huber_vs_refnet_e5`: 1000-game screen was positive, clean follow-up was negative.
- Imported binpack rows with `score=0`: many are not equal positions; do not train on them as CP labels.
- Long training based only on better MAE/sign numbers without replay and SPRT.

## Decision Rules

After each candidate:

1. Compare static metrics against baseline on:
   - current depth-16 bucket validation;
   - self-play depth-12 validation;
   - Lichess eval validation;
   - binpack validation.
2. Reject immediately if metrics are distorted or worse across important sources.
3. While replay is WIP, skip replay gates for NNUE promotion decisions.
4. Start SPRT only if source-separated metrics are acceptable.
5. Promote only if SPRT is meaningfully positive.

Neutral candidates are not promoted. They are kept as experiment artifacts.
Once replay is stable again, replay gates return as a required safety check.

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
