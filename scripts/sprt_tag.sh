#!/usr/bin/env bash

set -euo pipefail

usage()
{
    cat >&2 <<'EOF'
usage: ./scripts/sprt_tag.sh START_COMMIT

Run distributed SPRTs for each first-parent commit after START_COMMIT through
the current branch tip. Each candidate commit is recreated with its aggregate
Elo and LLR appended to the subject. Worker-local engine binaries are moved
from the old commit hash to the rewritten hash, and the branch is force-pushed
with an exact lease only after the complete chain succeeds. If a candidate's
upper 95% Elo bound is below zero, later commits are replayed without SPRT or
result labels.
EOF
}

die()
{
    printf 'sprt_tag: %s\n' "$*" >&2
    exit 1
}

release_lock()
{
    if test "${LOCK_HELD:-0}" = 1 && test -d "${LOCK_DIR:-}"; then
        if test "$(sed -n '1p' "$LOCK_DIR/pid" 2>/dev/null || true)" = "$$"; then
            rm -rf "$LOCK_DIR"
        fi
    fi
}

acquire_lock()
{
    local owner

    LOCK_DIR=$1
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        owner=$(sed -n '1p' "$LOCK_DIR/pid" 2>/dev/null || true)
        if test -n "$owner" && kill -0 "$owner" 2>/dev/null; then
            die "another sprt_tag.sh is running (pid=$owner)"
        fi
        test -n "$owner" || die "SPRT lock is held while its owner is initializing: $LOCK_DIR"
        printf 'Removing stale SPRT lock from pid %s\n' "$owner"
        rm -rf "$LOCK_DIR"
        mkdir "$LOCK_DIR" || die "cannot acquire SPRT lock: $LOCK_DIR"
    fi
    printf '%s\n' "$$" >"$LOCK_DIR/pid"
    LOCK_HELD=1
    trap release_lock EXIT
}

require_command()
{
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

write_value()
{
    local path=$1
    local value=$2
    printf '%s\n' "$value" >"$path.tmp"
    mv -f "$path.tmp" "$path"
}

read_value()
{
    local path=$1
    test -f "$path" || die "missing state file: $path"
    sed -n '1p' "$path"
}

short_hash()
{
    git rev-parse "$1^{commit}" | cut -c1-7
}

commit_tree()
{
    git rev-parse "$1^{tree}"
}

normalize_number()
{
    local value=$1
    case "$value" in
        +*|-*) printf '%s\n' "$value" ;;
        *) printf '+%s\n' "$value" ;;
    esac
}

validate_number()
{
    local label=$1
    local value=$2
    [[ "$value" =~ ^[+-]?[0-9]+([.][0-9]+)?$ ]] ||
        die "invalid $label value from Forge: $value"
}

significantly_worse()
{
    local elo=$1
    local ci=$2
    awk -v elo="$elo" -v ci="$ci" 'BEGIN { exit !((elo + ci) < 0.0) }'
}

strip_result_suffix()
{
    sed -E 's/[[:space:]]+elo[+-][0-9]+([.][0-9]+)?,[[:space:]]*llr[+-][0-9]+([.][0-9]+)?$//'
}

status_value()
{
    local json=$1
    local prefix=$2
    "$JQ" -er --arg prefix "$prefix" '
        [.display.fields[]? | select(startswith($prefix))][0]
        | split("=")[1]
    ' <<<"$json"
}

forge_status()
{
    "$FORGE" status "$1" --json
}

wait_for_existing_run()
{
    local run=$1
    local status
    local state

    status=$(forge_status "$run") || die "cannot read Forge run: $run"
    state=$("$JQ" -r '.state // ""' <<<"$status")
    case "$state" in
        done|ok|stopped) return ;;
        failed) die "Forge run failed: $run" ;;
    esac

    "$FORGE" resume "$run" --notify-command "$NOTIFY_COMMAND"

    while true; do
        status=$(forge_status "$run") || die "cannot read resumed Forge run: $run"
        state=$("$JQ" -r '.state // ""' <<<"$status")
        case "$state" in
            done|ok|stopped) return ;;
            failed) die "Forge run failed after resume: $run" ;;
        esac
        sleep "${SPRT_TAG_POLL_SECONDS:-30}"
    done
}

run_sprt()
{
    local index=$1
    local reference_hash=$2
    local candidate_hash=$3
    local run
    local run_index=${index//\//-}
    local status
    local state
    local elo
    local ci
    local llr
    local rc

    if test -f "$STATE_DIR/active_run"; then
        run=$(read_value "$STATE_DIR/active_run")
        test "$(read_value "$STATE_DIR/active_candidate")" = "$candidate_hash" ||
            die "state belongs to another active candidate"
        printf 'Resuming Forge run %s\n' "$run"
        wait_for_existing_run "$run"
    else
        run="sprt-tag-${candidate_hash}-vs-${reference_hash}-$(date +%Y%m%d-%H%M%S)-${run_index}"
        write_value "$STATE_DIR/active_run" "$run"
        write_value "$STATE_DIR/active_candidate" "$candidate_hash"

        printf 'SPRT %s: enyo_%s vs enyo_%s\n' "$index" "$candidate_hash" "$reference_hash"
        set +e
        "$FORGE" sprt \
            --run "$run" \
            --wait \
            --notify-command "$NOTIFY_COMMAND" \
            --comment "sprt-tag enyo_$candidate_hash vs enyo_$reference_hash" \
            --book "$BOOK" \
            --candidate "~/assets/engines/enyo_$candidate_hash" \
            --candidate-uci "nnue_file=~/assets/nets/default.net" \
            --concurrency "$CONCURRENCY" \
            --games "$GAMES" \
            --reference "~/assets/engines/enyo_$reference_hash" \
            --reference-uci "nnue_file=~/assets/nets/default.net" \
            --restart on \
            --shards "$SHARDS" \
            --tc "$TC" \
            --threads "$THREADS"
        rc=$?
        set -e

        if (( rc != 0 )); then
            status=$(forge_status "$run" 2>/dev/null || true)
            state=$("$JQ" -r '.state // ""' <<<"${status:-{}}" 2>/dev/null || true)
            case "$state" in
                done|ok|stopped) ;;
                *) die "Forge run did not complete successfully: $run (rc=$rc)" ;;
            esac
        fi
    fi

    status=$(forge_status "$run") || die "cannot read final Forge result: $run"
    state=$("$JQ" -r '.state // ""' <<<"$status")
    case "$state" in
        done|ok|stopped) ;;
        *) die "Forge run is not complete: $run (state=${state:-unknown})" ;;
    esac

    elo=$(status_value "$status" 'elo=') || die "Forge result has no aggregate Elo: $run"
    ci=$(status_value "$status" 'ci=') || die "Forge result has no aggregate Elo CI: $run"
    llr=$(status_value "$status" 'llr=') || die "Forge result has no aggregate LLR: $run"
    llr=${llr%%/*}
    validate_number elo "$elo"
    validate_number ci "$ci"
    validate_number llr "$llr"
    elo=$(normalize_number "$elo")
    llr=$(normalize_number "$llr")

    write_value "$STATE_DIR/result_old" "$candidate_hash"
    write_value "$STATE_DIR/result_elo" "$elo"
    write_value "$STATE_DIR/result_ci" "$ci"
    write_value "$STATE_DIR/result_llr" "$llr"
    rm -f "$STATE_DIR/active_run" "$STATE_DIR/active_candidate"

    printf '%s\t%s\t%s\n' "$elo" "$ci" "$llr"
}

rewrite_commit()
{
    local old_commit=$1
    local elo=$2
    local llr=$3
    local expected_parent
    local current
    local old_tree
    local subject
    local body
    local author_name
    local author_email
    local author_date
    local new_commit
    local new_tree
    local message_file="$STATE_DIR/commit-message"

    expected_parent=$(read_value "$STATE_DIR/result_parent")
    current=$(git -C "$WORKTREE" rev-parse HEAD)
    old_tree=$(commit_tree "$old_commit")

    if test "$current" != "$expected_parent" &&
       test "$(git -C "$WORKTREE" rev-parse HEAD^)" = "$expected_parent" &&
       test "$(git -C "$WORKTREE" rev-parse HEAD^{tree})" = "$old_tree"; then
        printf '%s\n' "$current"
        return
    fi
    test "$current" = "$expected_parent" ||
        die "rewrite worktree moved unexpectedly: $current"

    git -C "$WORKTREE" cherry-pick --no-commit "$old_commit" ||
        die "cannot replay $old_commit; conflict left in $WORKTREE"

    subject=$(git show -s --format=%s "$old_commit" | strip_result_suffix)
    body=$(git show -s --format=%b "$old_commit")
    author_name=$(git show -s --format=%an "$old_commit")
    author_email=$(git show -s --format=%ae "$old_commit")
    author_date=$(git show -s --format=%aI "$old_commit")

    {
        printf '%s elo%s, llr%s\n' "$subject" "$elo" "$llr"
        if test -n "$body"; then
            printf '\n%s\n' "$body"
        fi
    } >"$message_file"

    GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_AUTHOR_DATE="$author_date" \
        git -C "$WORKTREE" -c commit.gpgsign=false commit --no-verify -F "$message_file" >/dev/null

    new_commit=$(git -C "$WORKTREE" rev-parse HEAD)
    new_tree=$(git -C "$WORKTREE" rev-parse HEAD^{tree})
    test "$new_tree" = "$old_tree" ||
        die "rewritten tree differs for $old_commit: $new_tree != $old_tree"
    printf '%s\n' "$new_commit"
}

replay_unlabeled_commit()
{
    local old_commit=$1
    local expected_parent
    local current
    local old_tree
    local author_name
    local author_email
    local author_date
    local new_commit
    local new_tree
    local message_file="$STATE_DIR/commit-message"

    expected_parent=$(read_value "$STATE_DIR/result_parent")
    current=$(git -C "$WORKTREE" rev-parse HEAD)
    old_tree=$(commit_tree "$old_commit")

    if test "$current" != "$expected_parent" &&
       test "$(git -C "$WORKTREE" rev-parse HEAD^)" = "$expected_parent" &&
       test "$(git -C "$WORKTREE" rev-parse HEAD^{tree})" = "$old_tree"; then
        printf '%s\n' "$current"
        return
    fi
    test "$current" = "$expected_parent" ||
        die "rewrite worktree moved unexpectedly: $current"

    git -C "$WORKTREE" cherry-pick --no-commit "$old_commit" ||
        die "cannot replay $old_commit; conflict left in $WORKTREE"
    git show -s --format=%B "$old_commit" >"$message_file"
    author_name=$(git show -s --format=%an "$old_commit")
    author_email=$(git show -s --format=%ae "$old_commit")
    author_date=$(git show -s --format=%aI "$old_commit")

    GIT_AUTHOR_NAME="$author_name" \
    GIT_AUTHOR_EMAIL="$author_email" \
    GIT_AUTHOR_DATE="$author_date" \
        git -C "$WORKTREE" -c commit.gpgsign=false commit --no-verify -F "$message_file" >/dev/null

    new_commit=$(git -C "$WORKTREE" rev-parse HEAD)
    new_tree=$(git -C "$WORKTREE" rev-parse HEAD^{tree})
    test "$new_tree" = "$old_tree" ||
        die "replayed tree differs for $old_commit: $new_tree != $old_tree"
    printf '%s\n' "$new_commit"
}

move_worker_binary()
{
    local old_hash=$1
    local new_hash=$2
    local command

    test "$old_hash" != "$new_hash" || return
    command="set -eu
src=\"\$HOME/assets/engines/enyo_$old_hash\"
dst=\"\$HOME/assets/engines/enyo_$new_hash\"
if ! test -x \"\$src\"; then
    if test -x \"\$dst\"; then
        ln -sfn \"\$dst\" \"\$HOME/assets/engines/candidate\"
    else
        echo \"engine not present on this worker; skipping: \$src\"
    fi
    exit 0
fi
if test -e \"\$dst\"; then
    cmp -s \"\$src\" \"\$dst\" || { echo \"different engine already exists: \$dst\" >&2; exit 1; }
    rm -f \"\$src\"
else
    mv \"\$src\" \"\$dst\"
fi
chmod +x \"\$dst\"
ln -sfn \"\$dst\" \"\$HOME/assets/engines/candidate\""

    "$FORGE" sh "$command"
}

finalize_pending_move()
{
    local index=$1
    local old_hash
    local new_hash
    local old_commit
    local elo
    local ci
    local llr
    local stop

    old_hash=$(read_value "$STATE_DIR/pending_old_hash")
    new_hash=$(read_value "$STATE_DIR/pending_new_hash")
    old_commit=$(read_value "$STATE_DIR/pending_old_commit")
    elo=$(read_value "$STATE_DIR/pending_elo")
    ci=$(read_value "$STATE_DIR/pending_ci")
    llr=$(read_value "$STATE_DIR/pending_llr")
    stop=$(read_value "$STATE_DIR/pending_stop")

    printf 'Moving worker engines enyo_%s -> enyo_%s\n' "$old_hash" "$new_hash"
    move_worker_binary "$old_hash" "$new_hash"

    if ! grep -q "^$old_commit"$'\t' "$STATE_DIR/mapping.tsv"; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$old_commit" \
            "$(git -C "$WORKTREE" rev-parse HEAD)" "$elo" "$ci" "$llr" >>"$STATE_DIR/mapping.tsv"
    fi
    write_value "$STATE_DIR/reference_hash" "$new_hash"
    write_value "$STATE_DIR/next_index" "$((index + 1))"
    if test "$stop" = 1; then
        write_value "$STATE_DIR/stop_labeling" 1
    fi
    rm -f "$STATE_DIR"/pending_* "$STATE_DIR"/result_*
}

remote_head()
{
    git ls-remote --heads "$1" "refs/heads/$2" | sed -n '1{s/[[:space:]].*//;p;}'
}

resolve_notify_command()
{
    local forge_path
    local target
    local root

    if test -n "${SPRT_TAG_NOTIFY_COMMAND:-}"; then
        printf '%s\n' "$SPRT_TAG_NOTIFY_COMMAND"
        return
    fi
    if test -n "${FORGE_NOTIFY_COMMAND:-}"; then
        printf '%s\n' "$FORGE_NOTIFY_COMMAND"
        return
    fi

    forge_path=$(command -v "$FORGE")
    target=$forge_path
    if test -L "$forge_path"; then
        target=$(readlink "$forge_path")
        case "$target" in
            /*) ;;
            *) target="$(dirname "$forge_path")/$target" ;;
        esac
    fi
    root=$(cd "$(dirname "$target")" && pwd)
    if test -x "$root/scripts/forge_event_ntfy.sh"; then
        printf '%s\n' "$root/scripts/forge_event_ntfy.sh"
        return
    fi
    die "set SPRT_TAG_NOTIFY_COMMAND or FORGE_NOTIFY_COMMAND"
}

restore_worktree()
{
    local restore_commit
    local mapped_commit

    restore_commit=$(read_value "$STATE_DIR/start")
    if test -f "$STATE_DIR/pending_new_commit"; then
        restore_commit=$(read_value "$STATE_DIR/pending_new_commit")
    elif test -s "$STATE_DIR/mapping.tsv"; then
        mapped_commit=$(awk -F '\t' 'NF >= 2 { value=$2 } END { print value }' "$STATE_DIR/mapping.tsv")
        if test -n "$mapped_commit"; then
            restore_commit=$mapped_commit
        fi
    elif test -f "$STATE_DIR/result_parent"; then
        restore_commit=$(read_value "$STATE_DIR/result_parent")
    fi

    git cat-file -e "$restore_commit^{commit}" 2>/dev/null ||
        die "cannot restore missing rewrite commit: $restore_commit"
    printf 'Restoring missing rewrite worktree at %s\n' "$WORKTREE"
    git worktree prune
    mkdir -p "$(dirname "$WORKTREE")"
    git worktree add --detach "$WORKTREE" "$restore_commit" >/dev/null
}

if (( $# != 1 )); then
    usage
    exit 2
fi

START_ARG=$1

interrupted()
{
    printf '\nInterrupted. Nothing needs cleanup; resume with:\n  ./scripts/sprt_tag.sh %s\n' "$START_ARG" >&2
    exit 130
}

trap interrupted INT TERM

FORGE=${SPRT_TAG_FORGE:-forge}
JQ=${SPRT_TAG_JQ:-jq}
GAMES=${SPRT_TAG_GAMES:-1000}
SHARDS=${SPRT_TAG_SHARDS:-24}
TEST_ALL=${SPRT_TAG_TEST_ALL:-0}
CONCURRENCY=${SPRT_TAG_CONCURRENCY:-1}
THREADS=${SPRT_TAG_THREADS:-1}
TC=${SPRT_TAG_TC:-10+0.1}
BOOK=${SPRT_TAG_BOOK:-~/assets/books/AntiDraw_V2.1/WOMP_Openings_V1/WOMP_V1_+150_+159/WOMP_V1_6mvs_big_+140_+169.epd}

require_command git
require_command "$FORGE"
require_command "$JQ"

REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || die "not inside a Git checkout"
cd "$REPO_ROOT"
EARLY_GIT_DIR=$(git rev-parse --absolute-git-dir)
acquire_lock "$EARLY_GIT_DIR/sprt-tag.lock"

BRANCH=$(git symbolic-ref --quiet --short HEAD || true)
if test -z "$BRANCH"; then
    DETACHED_START=$(git rev-parse --verify "$START_ARG^{commit}") || die "invalid start commit: $START_ARG"
    DETACHED_GIT_DIR=$(git rev-parse --absolute-git-dir)
    DETACHED_STATE_ROOT=${SPRT_TAG_STATE_ROOT:-$DETACHED_GIT_DIR/sprt-tag}
    SAVED_BRANCH=
    for saved_state in "$DETACHED_STATE_ROOT"/*; do
        test -d "$saved_state" || continue
        test -f "$saved_state/start" || continue
        test "$(sed -n '1p' "$saved_state/start")" = "$DETACHED_START" || continue
        test -z "$SAVED_BRANCH" || die "multiple resumable states match $START_ARG"
        SAVED_BRANCH=$(sed -n '1p' "$saved_state/branch")
    done
    test -n "$SAVED_BRANCH" || die "detached HEAD has no matching resumable state"
    printf 'Restoring primary checkout to %s\n' "$SAVED_BRANCH"
    git switch "$SAVED_BRANCH" >/dev/null
    BRANCH=$SAVED_BRANCH
fi
UPSTREAM=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) ||
    die "current branch has no upstream"
case "$UPSTREAM" in
    */*) ;;
    *) die "cannot parse upstream: $UPSTREAM" ;;
esac
REMOTE=${UPSTREAM%%/*}
REMOTE_BRANCH=${UPSTREAM#*/}

git diff --quiet || die "tracked working tree changes must be committed or stashed"
git diff --cached --quiet || die "staged changes must be committed or stashed"

USER_NAME=$(git config user.name || true)
USER_EMAIL=$(git config user.email || true)
test -n "$USER_NAME" && test -n "$USER_EMAIL" || die "Git user.name and user.email must be configured"
if [[ "$USER_NAME $USER_EMAIL" =~ ([Bb]ot|[Cc]odex|[Cc]opilot|[Cc]laude|github-actions) ]]; then
    die "bot Git identity is not allowed: $USER_NAME <$USER_EMAIL>"
fi

START=$(git rev-parse --verify "$START_ARG^{commit}") || die "invalid start commit: $START_ARG"
ORIGINAL_TIP=$(git rev-parse HEAD)
git merge-base --is-ancestor "$START" "$ORIGINAL_TIP" ||
    die "$START_ARG is not an ancestor of HEAD"

git fetch "$REMOTE"
REMOTE_TIP=$(remote_head "$REMOTE" "$REMOTE_BRANCH")
test -n "$REMOTE_TIP" || die "remote branch not found: $UPSTREAM"
if test "$REMOTE_TIP" != "$ORIGINAL_TIP"; then
    if git merge-base --is-ancestor "$REMOTE_TIP" "$ORIGINAL_TIP"; then
        git push "$REMOTE" "HEAD:refs/heads/$REMOTE_BRANCH"
        REMOTE_TIP=$ORIGINAL_TIP
    else
        die "local $BRANCH and $UPSTREAM are not synchronized"
    fi
fi

while IFS= read -r commit; do
    test "$(git rev-list --parents -n 1 "$commit" | wc -w | tr -d ' ')" = 2 ||
        die "merge commits are not supported: $commit"
done < <(git rev-list --reverse --first-parent "$START..$ORIGINAL_TIP")

NOTIFY_COMMAND=$(resolve_notify_command)
test -x "${NOTIFY_COMMAND%% *}" || die "notification hook is not executable: $NOTIFY_COMMAND"

GIT_DIR=$(git rev-parse --absolute-git-dir)
START_SHORT=$(short_hash "$START")
TIP_SHORT=$(short_hash "$ORIGINAL_TIP")
STATE_KEY=$(printf '%s-%s-%s' "$BRANCH" "$START_SHORT" "$TIP_SHORT" | tr -c 'A-Za-z0-9._-' '_')
STATE_ROOT=${SPRT_TAG_STATE_ROOT:-$GIT_DIR/sprt-tag}
STATE_DIR=$STATE_ROOT/$STATE_KEY
WORK_ROOT=${SPRT_TAG_WORK_ROOT:-$HOME/tmp}
WORKTREE=$WORK_ROOT/enyo-sprt-tag-$STATE_KEY

mkdir -p "$STATE_ROOT" "$WORK_ROOT"
if test -d "$STATE_DIR"; then
    test "$(read_value "$STATE_DIR/original_tip")" = "$ORIGINAL_TIP" || die "state tip mismatch"
    test "$(read_value "$STATE_DIR/start")" = "$START" || die "state start mismatch"
    WORKTREE=$(read_value "$STATE_DIR/worktree")
    if ! test -d "$WORKTREE"; then
        restore_worktree
    fi
    printf 'Resuming state from %s\n' "$STATE_DIR"
else
    mkdir -p "$STATE_DIR"
    write_value "$STATE_DIR/start" "$START"
    write_value "$STATE_DIR/original_tip" "$ORIGINAL_TIP"
    write_value "$STATE_DIR/original_remote_tip" "$REMOTE_TIP"
    write_value "$STATE_DIR/branch" "$BRANCH"
    write_value "$STATE_DIR/upstream" "$UPSTREAM"
    write_value "$STATE_DIR/worktree" "$WORKTREE"
    write_value "$STATE_DIR/next_index" 0
    write_value "$STATE_DIR/reference_hash" "$START_SHORT"
    : >"$STATE_DIR/mapping.tsv"
    git rev-list --reverse --first-parent "$START..$ORIGINAL_TIP" >"$STATE_DIR/commits"
    test ! -e "$WORKTREE" || die "rewrite worktree path already exists: $WORKTREE"
    git worktree add --detach "$WORKTREE" "$START" >/dev/null
fi

TOTAL=$(wc -l <"$STATE_DIR/commits" | tr -d ' ')
INDEX=$(read_value "$STATE_DIR/next_index")

while (( INDEX < TOTAL )); do
    if test -f "$STATE_DIR/pending_old_hash"; then
        PENDING_INDEX=$(read_value "$STATE_DIR/pending_index")
        finalize_pending_move "$PENDING_INDEX"
        INDEX=$(read_value "$STATE_DIR/next_index")
        continue
    fi

    OLD_COMMIT=$(sed -n "$((INDEX + 1))p" "$STATE_DIR/commits")
    OLD_PARENT=$(git rev-parse "$OLD_COMMIT^")
    OLD_HASH=$(short_hash "$OLD_COMMIT")

    if test "$(commit_tree "$OLD_COMMIT")" = "$(commit_tree "$OLD_PARENT")"; then
        printf 'Skipping empty commit %s %s\n' "$OLD_HASH" "$(git show -s --format=%s "$OLD_COMMIT")"
        printf '%s\t%s\tempty\tempty\n' "$OLD_COMMIT" "$(git -C "$WORKTREE" rev-parse HEAD)" >>"$STATE_DIR/mapping.tsv"
        INDEX=$((INDEX + 1))
        write_value "$STATE_DIR/next_index" "$INDEX"
        continue
    fi

    LABEL_CURRENT=1
    STOP_AFTER=0
    if test "$TEST_ALL" != 1 && test -f "$STATE_DIR/stop_labeling"; then
        LABEL_CURRENT=0
        STOP_AFTER=1
        ELO=unlabeled
        CI=unlabeled
        LLR=unlabeled
        printf 'Replaying without SPRT after significant regression: %s %s\n' \
            "$OLD_HASH" "$(git show -s --format=%s "$OLD_COMMIT")"
    elif test -f "$STATE_DIR/result_old"; then
        test "$(read_value "$STATE_DIR/result_old")" = "$OLD_HASH" ||
            die "saved result belongs to another commit"
        ELO=$(read_value "$STATE_DIR/result_elo")
        CI=$(read_value "$STATE_DIR/result_ci")
        LLR=$(read_value "$STATE_DIR/result_llr")
    else
        REFERENCE_HASH=$(read_value "$STATE_DIR/reference_hash")
        run_sprt "$((INDEX + 1))/$TOTAL" "$REFERENCE_HASH" "$OLD_HASH"
        ELO=$(read_value "$STATE_DIR/result_elo")
        CI=$(read_value "$STATE_DIR/result_ci")
        LLR=$(read_value "$STATE_DIR/result_llr")
    fi

    if ! git symbolic-ref --quiet --short HEAD >/dev/null; then
        printf 'Restoring primary checkout to %s after worker deployment\n' "$BRANCH"
        git switch "$BRANCH" >/dev/null
    fi

    if test "$TEST_ALL" != 1 && test "$LABEL_CURRENT" = 1 && significantly_worse "$ELO" "$CI"; then
        STOP_AFTER=1
        printf 'Significant regression at %s: elo=%s +/- %s; stopping later SPRTs\n' \
            "$OLD_HASH" "$ELO" "$CI"
    fi

    if ! test -f "$STATE_DIR/result_parent"; then
        write_value "$STATE_DIR/result_parent" "$(git -C "$WORKTREE" rev-parse HEAD)"
    fi
    if test "$LABEL_CURRENT" = 1; then
        NEW_COMMIT=$(rewrite_commit "$OLD_COMMIT" "$ELO" "$LLR")
    else
        NEW_COMMIT=$(replay_unlabeled_commit "$OLD_COMMIT")
    fi
    NEW_HASH=$(short_hash "$NEW_COMMIT")

    write_value "$STATE_DIR/pending_old_hash" "$OLD_HASH"
    write_value "$STATE_DIR/pending_new_hash" "$NEW_HASH"
    write_value "$STATE_DIR/pending_new_commit" "$NEW_COMMIT"
    write_value "$STATE_DIR/pending_old_commit" "$OLD_COMMIT"
    write_value "$STATE_DIR/pending_index" "$INDEX"
    write_value "$STATE_DIR/pending_elo" "$ELO"
    write_value "$STATE_DIR/pending_ci" "$CI"
    write_value "$STATE_DIR/pending_llr" "$LLR"
    write_value "$STATE_DIR/pending_stop" "$STOP_AFTER"
    finalize_pending_move "$INDEX"
    INDEX=$(read_value "$STATE_DIR/next_index")
done

NEW_TIP=$(git -C "$WORKTREE" rev-parse HEAD)
test "$(commit_tree "$NEW_TIP")" = "$(commit_tree "$ORIGINAL_TIP")" ||
    die "final rewritten tree differs from the original tip"

EXPECTED_REMOTE=$(read_value "$STATE_DIR/original_remote_tip")
CURRENT_REMOTE=$(remote_head "$REMOTE" "$REMOTE_BRANCH")
test "$CURRENT_REMOTE" = "$EXPECTED_REMOTE" ||
    die "$UPSTREAM moved during SPRT; refusing to rewrite it"

printf 'Publishing rewritten %s: %s -> %s\n' "$BRANCH" "$(short_hash "$ORIGINAL_TIP")" "$(short_hash "$NEW_TIP")"
if test "$(git symbolic-ref --quiet --short HEAD || true)" != "$BRANCH"; then
    git switch "$BRANCH" >/dev/null
fi
git push \
    --force-with-lease="refs/heads/$REMOTE_BRANCH:$EXPECTED_REMOTE" \
    "$REMOTE" \
    "$NEW_TIP:refs/heads/$REMOTE_BRANCH"
git reset --hard "$NEW_TIP"
git worktree remove --force "$WORKTREE"
rm -rf "$STATE_DIR"

printf 'Done: %s now points to %s\n' "$BRANCH" "$(short_hash HEAD)"
