#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"

main_ref="${MAIN_REF:-main}"
build_type="${BUILD_TYPE:-Release}"
jobs="${JOBS:-}"
default_assets_dir="$repo_root/../assets/engines"
if [[ ! -d "$default_assets_dir" ]]; then
    git_common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"
    if [[ "$(basename "$git_common_dir")" == ".git" ]]; then
        default_assets_dir="$(dirname "$git_common_dir")/../assets/engines"
    fi
fi
assets_dir="${ASSETS_ENGINES_DIR:-$default_assets_dir}"

assets_dir="$(cd "$assets_dir" && pwd -P)"
old_reference="$(readlink "$assets_dir/reference" || true)"

if [[ -z "$old_reference" ]]; then
    echo "ERROR: $assets_dir/reference is missing or is not a symlink" >&2
    exit 1
fi

if [[ "$old_reference" == /* ]]; then
    old_reference_path="$old_reference"
else
    old_reference_path="$assets_dir/$old_reference"
fi

if [[ ! -e "$old_reference_path" ]]; then
    echo "ERROR: old reference target does not exist: $old_reference_path" >&2
    exit 1
fi

main_commit="$(git -C "$repo_root" rev-parse --verify "$main_ref^{commit}")"
short_hash="$(git -C "$repo_root" rev-parse --short "$main_commit")"
dest="$assets_dir/enyo_$short_hash"

if [[ -e "$dest" ]]; then
    echo "ERROR: destination already exists: $dest" >&2
    echo "Remove it first if this rebuild is intentional." >&2
    exit 1
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/enyo-reference-${short_hash}.XXXXXX")"
worktree="$tmp_root/src"
build_dir="$tmp_root/build"

cleanup() {
    git -C "$repo_root" worktree remove --force "$worktree" >/dev/null 2>&1 || true
    rm -rf "$tmp_root"
}
trap cleanup EXIT

git -C "$repo_root" worktree add --detach "$worktree" "$main_commit"

if [[ -n "$(git -C "$worktree" status --porcelain)" ]]; then
    echo "ERROR: clean worktree is unexpectedly dirty" >&2
    exit 1
fi

if [[ -z "$jobs" ]]; then
    jobs="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
fi

cmake -S "$worktree" -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type"
cmake --build "$build_dir" --target enyo -j "$jobs"

install -m 755 "$build_dir/enyo" "$dest"

(
    cd "$assets_dir"
    ln -sfn "$old_reference" candidate
    ln -sfn "enyo_$short_hash" reference
)

echo "candidate -> $old_reference"
echo "reference -> enyo_$short_hash"
printf 'uci\nquit\n' | "$assets_dir/reference" | grep -E '^(id name|option name root_repetition_contempt|uciok)'
