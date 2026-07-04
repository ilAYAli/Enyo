#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)
STATE="$ROOT/spsa/state.json"
TARGET=${1:-3000}

die() {
    echo "error: $*" >&2
    exit 1
}

case "$TARGET" in
    ''|*[!0-9]*) die "target iteration must be a positive integer" ;;
esac
(( TARGET > 0 )) || die "target iteration must be positive"

cd "$ROOT"
[[ $(git branch --show-current) == main ]] || die "run this script from branch main"
git ls-files --error-unmatch spsa/state.json >/dev/null 2>&1 ||
    die "spsa/state.json must be tracked"

unexpected=$(git diff HEAD --name-only -- . ':!spsa/state.json')
[[ -z $unexpected ]] || {
    echo "error: unrelated tracked changes must be committed first:" >&2
    echo "$unexpected" >&2
    exit 1
}

git config user.name "Petter Wahlman"
git config user.email "petter@wahlman.no"

checkpoint_state() {
    local iteration branch
    iteration=$(jq -er '.k' "$STATE")
    [[ $(jq -r '.batch.complete // true' "$STATE") == true ]] ||
        die "refusing to checkpoint incomplete iteration $iteration"

    if git diff HEAD --quiet -- spsa/state.json; then
        echo "SPSA iteration $iteration is already committed"
        return
    fi

    branch="feature/spsa-k${iteration}-checkpoint"
    git show-ref --verify --quiet "refs/heads/$branch" &&
        die "branch already exists: $branch"

    git fetch origin
    git merge --ff-only origin/main
    git switch -c "$branch"
    git add spsa/state.json
    git commit -m "chore: checkpoint SPSA iteration $iteration"
    git fetch origin
    git rebase origin/main
    git switch main
    git merge --ff-only "$branch"
    git branch -d "$branch"
    git push origin main
}

current=$(jq -er '.k' "$STATE")
complete=$(jq -r '.batch.complete // true' "$STATE")
stored_target=$(jq -er '.batch.target_k // .k' "$STATE")
(( TARGET >= current )) ||
    die "target $TARGET is below current iteration $current"

if [[ $complete == true ]]; then
    checkpoint_state
elif (( stored_target != TARGET )); then
    die "unfinished batch targets $stored_target, not requested target $TARGET"
fi

current=$(jq -er '.k' "$STATE")
complete=$(jq -r '.batch.complete // true' "$STATE")
if (( current < TARGET )); then
    echo "Tuning SPSA from iteration $current to $TARGET"
    "$ROOT/spsa/tune.py" --iterations "$((TARGET - current))"
elif [[ $complete != true ]]; then
    echo "Finalizing completed batch at iteration $TARGET"
    "$ROOT/spsa/tune.py" --iterations 1
else
    echo "SPSA is already at iteration $TARGET"
fi

current=$(jq -er '.k' "$STATE")
complete=$(jq -r '.batch.complete // false' "$STATE")
(( current == TARGET )) || die "tuner stopped at iteration $current, expected $TARGET"
[[ $complete == true ]] || die "SPSA batch at iteration $TARGET is incomplete"

checkpoint_state

run="spsa-k${TARGET}-vs-promoted-defaults"
echo "Launching detached 1000-game SPRT: $run"
"$ROOT/spsa/sprt.py" --run "$run"
