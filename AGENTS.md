# NNUE Development Agents

This file defines three separate agent roles. Do not merge their
responsibilities.

## 1. Coding Agent

Owns training/tooling changes only.

### Responsibilities

- All code that affects the hot path must be very fast
- All features should be implemented in a separate feature branch
- I don't want merge commits
- When a feature is complete, run relevant tests before merging with main.
- After merging with main, make a new candidate by running ./scripts/make_candidate.sh
- Keep changes narrow and tied to the active hypothesis in
  the workflow.

### Rules

- Change one hypothesis at a time: target construction, loss/objective,
  architecture, or data.


## 2. Validation Agent

Owns NNUE verification only.

### Responsibilities

- Run parity checks, exported gates, engine-side gates, replay/failure suites,
  and game smokes.
- Report exact numbers and classify the result.
- Do not modify source except for generated validation outputs explicitly
  requested by the Coding Agent.
- Do not commit.

1. Replay/failure suite
   - Useful as a rejection filter.
   - Positive replay is not enough for promotion.

2. Game test
   - Run an early 200-300 game smoke before a full 1000-game SPRT.
   - If three consecutive candidates from the same objective family fail the
     smoke, close that family and change the failure theory.

For architecture changes, run incremental-vs-refresh accumulator tests and NPS
checks before training.

### Interpretation

Consider a candidate only if:

- exported gates pass;
- replay has no unexplained tail regression;
- a smoke game test is at least neutral-positive.

## 3. Git / Release Agent

Owns git operations only.

### Responsibilities

- Create branches, stage files, commit, rebase, merge, tag, and push when
  explicitly instructed.
- Enforce clean history and correct commit identity.
- Never resolve source conflicts without handing back to the Coding Agent.

### Rules

- Never commit directly to `main`.
- Never create merge commits unless the user explicitly asks.
- Rebase feature branches onto `main`; merge to `main` with `--ff-only`.
- Never rewrite `main`.
- Never use bot authors or AI co-author trailers.
- Verify commit identity before committing:

```sh
git config user.name "Petter Wahlman"
git config user.email "petter@wahlman.no"
```

- Stage only files that belong to the requested change.
- Keep one feature, fix, or experiment in one logical commit whenever
  practical, so it is easy to revert.
- Refactor or cleanup work found while developing should preferably be amended
  or squashed into the commit that introduced the code. Use a separate refactor
  commit only when the refactor is independently useful and does not change
  behavior.
- Squash local fixup churn before merge.
- Do not commit unrelated NNUE run artifacts.
- If a commit message is needed, request it from the Coding Agent.

## Shared Documentation Rules

- Update `IMPROVEMENT_PLAN.md` only for durable conclusions:
  - a lane is closed;
  - a new blocker is identified;
  - a tool/gate was corrected;
  - the next useful action changes.
- Keep documents practical: what to change, what to run, how to judge it, stop
  criteria.
- Do not add machine-specific hostnames, local paths, or private notes to public
  documentation.

## Shared Long-Run Rules

- MANDATORY NOTIFICATION RULE: every long-running Crucible or NNUE job must be
  launched with the repo notification hook already attached. For Crucible this
  means `--notify-command` or `CRUCIBLE_NOTIFY_COMMAND` at launch/resume/retry
  time. For NNUE this means `--event-command` or the configured build hook.
  There are no exceptions.
- Polling, watcher scripts, `sleep` loops, tmux tails, or manual status checks
  are not substitutes for launch-time notifications. They may only be used for
  diagnosis after the notification hook is already attached.
- Run long NNUE jobs in the appropriate tmux session.
- Every long-running task must notify `AI_stdin` on `done` and `fail` by using
  `~/scripts/notifai.sh` through `tools/events/nnue_event_ntfy.sh`.
  Do not post directly to `AI_stdin` unless `notifai.sh` is unavailable and the
  hook falls back to authenticated ntfy publishing.
- Notifications should report task, ETA, and project state. Avoid phase spam.
- Remove temporary tmux windows when the job is done.
- Do not leave nested shells in tmux panes.
- NNUE training is paused. Do not start a new run without an explicit instruction
  and a written hypothesis in `IMPROVEMENT_PLAN.md`.
- Engine stability work takes priority over NNUE research during the pause. Keep
  them separated: engine bugs go to the engine repo, NNUE tooling stays here.
