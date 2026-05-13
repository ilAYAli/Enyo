# Enyo NNUE Roadmap

This document tracks the process for building a competitive Enyo-owned NNUE.
It is a roadmap and status file, not the detailed training explainer. The
training mechanics and script details live in `README_nnue_training.html`.

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

Active run on `pwa-5090`:

```text
run:     /home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554
script:  /home/petter/tmp/run_nnue_bucket_d16_20260512.sh
tmux:    nnue_test
phase:   Stockfish depth-16 labeling
status:  510741 / 1000000 labels, about 51.1% at 2026-05-13 01:41 CEST
workers: 16 Stockfish processes, one thread each
```

Purpose of the active run:

- sample 1M unique Enyo-reached positions from the 20M self-play pool;
- balance the sample by absolute eval bucket;
- relabel with Stockfish depth 16;
- train two Huber-loss pilots from the current starting net;
- gate candidates with source-separated metrics and replay;
- start SPRT only if a candidate is worth testing.

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

## Active Goal: Higher-Quality Targets

The current hypothesis is that neutral results are caused more by target/data
quality than by trainer mechanics. The next useful experiment is therefore the
current depth-16 bucketed run.

Current run stages:

- [x] Sample 1M bucket-balanced unique rows.
- [ ] Relabel those rows with Stockfish depth 16.
- [ ] Audit target distribution and source-score disagreement.
- [ ] Pack the relabeled data.
- [ ] Evaluate the starting net on all validation sets.
- [ ] Train `d16_bucket_huber_lr1e7_e4`.
- [ ] Run static metrics and replay gates for `d16_bucket_huber_lr1e7_e4`.
- [ ] Start SPRT for `d16_bucket_huber_lr1e7_e4` if gates pass.
- [ ] If rejected, train `d16_bucket_huber_lr3e8_e4`.
- [ ] Run static metrics and replay gates for `d16_bucket_huber_lr3e8_e4`.
- [ ] Start SPRT for `d16_bucket_huber_lr3e8_e4` if gates pass.

Expected outputs:

```text
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/audit_targets.log
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed/
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/baseline/eval/
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/d16_bucket_huber_lr1e7_e4/model.nn
/home/petter/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/d16_bucket_huber_lr3e8_e4/model.nn
```

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

## If The Current Run Is Neutral

Do not keep trying random low-learning-rate variants of the same old mix.
The next steps should improve training signal quality:

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

Watch current training:

```sh
ssh -t petter@pwa-5090 tmux attach -t nnue_test
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
README_nnue_training.html
```
