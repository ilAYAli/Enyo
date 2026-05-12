# Enyo Development Instructions

## Scope Control

- Do not infer features.
- Do not redesign.
- Do not add behavior that was not explicitly requested.
- Before editing files, restate the exact requested behavior in one sentence.
- If the requested scope is unclear, ask before editing.
- Do not stop for input unless it is required to continue safely.

## Branch Rules

- New features must be implemented on a new git feature branch.
- Bug fixes may be implemented directly on `main` when the requested fix is narrow.
- A feature branch may be merged into `main` only after the feature has been proven correct.
- Commit completed work and push it to the private repository.

## Machines And Working Directories

- Primary remote test machine: `petter@pwa-5090`.
- Secondary deployment/test machine: `pwa-win`.
- Remote Enyo checkout: `~/code/cpp/chess/enyo`.
- Run tests and SPRTs on `pwa-5090` in the tmux sessions listed below.
- Do not detach from the tmux session after a test completes; leave the window open with the final output visible.
- Durable temporary files belong under `~/tmp/`.
- Remove temporary files when they are no longer needed.

## Tmux Sessions

- Prefer separate tmux sessions over multiple windows.
- Reuse existing tmux sessions instead of creating new ones.
- On `pwa-5090`, use these NNUE sessions:
  - `nnue_test`: active long-running NNUE training, replay gates, or SPRT validation.
  - `nnue_cmd`: short status checks, log inspection, and small helper commands.
- Do not create extra NNUE sessions or windows unless explicitly requested.
- Close idle sessions/windows that are no longer useful, but never kill an active training or test process unless requested.

## Notifications

- When attention is needed, send a ping notification to `https://ntfy.wahlman.no/ping`.
- Before sending authenticated notifications, run `source ~/.ntfy` to set `LICHESS_NTFY_AUTH`.
- SPRT notifications should use `https://ntfy.wahlman.no/sprt`.
- For long-running local or remote tasks, add a completion callback with:

```sh
~/scripts/notifai.sh "task finished: <short status>" <codex_tmux_session>
```

- `notifai.sh` sends a message into the Codex tmux session, so use it to report
  completion/failure without polling.
- If the Codex session is omitted, `notifai.sh` defaults to `codex_1`.
- Use both `notifai.sh` and ntfy for important long-running jobs: `notifai.sh`
  wakes the local agent loop, ntfy wakes the human.

## Development Location

- Development may happen locally or on `pwa-5090`.
- If work is done locally, pull and build it on `pwa-5090` before remote testing.
- If work is committed on `pwa-5090`, pull and build it locally before using it locally.
- Any runtime machine that did not create the commit must pull the commit and rebuild before it is used there.

## Build Candidate

Use the latest commit on `main` for reference/candidate binaries unless a feature branch is explicitly being tested.

Record the git hash used for each engine binary:

```sh
git rev-parse --short HEAD
```

Build the engine, then copy it to the assets engine directory using the hash in the filename:

```sh
cmake -B build
cmake --build build
cp ./build/enyo ../assets/engines/enyo_<githash>
```

## Test On pwa-5090

Use the appropriate tmux session for tests:

```sh
ssh -t petter@pwa-5090 tmux -2u new -As nnue_test
```

In the remote checkout:

```sh
cd ~/code/cpp/chess/enyo
```

If the work was implemented locally:

```sh
git pull
```

Build both the reference engine and the candidate engine before running SPRT.

Run SPRT tests with:

```sh
../assets/scripts/sprt --games 1000 --concurrency 8 --reference ../assets/engines/reference --candidate ./build/enyo --ntfy-url https://ntfy.wahlman.no/sprt
```

## After A Successful SPRT

A successful SPRT means the candidate accepted H1 or otherwise has an explicitly approved positive result.

After a successful SPRT:

1. Merge the proven change into `main` if it is not already on `main`.
2. Commit the change, including the SPRT score and Elo result in the commit message body.
3. Push `main` to the private repository.
4. Pull and rebuild on every runtime machine that did not create the commit:
   - local workstation, when the commit was created on `pwa-5090`
   - `pwa-5090`, when the commit was created locally
   - `pwa-win`
5. Copy the rebuilt engine to `../assets/engines/enyo_<githash>`.
6. If this engine is accepted as the new reference, update the reference engine according to the project script/workflow.
