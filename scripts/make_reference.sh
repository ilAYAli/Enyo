#!/usr/bin/env bash
set -euo pipefail

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(git -C "$script_dir" rev-parse --show-toplevel)"

cleanup_old=false
dry_run=false
keep_count="${KEEP_ENGINES:-10}"

usage() {
    cat <<'EOF'
Usage: scripts/make_reference.sh [--cleanup] [--dry-run] [--keep N]

Build clean main as assets/engines/enyo_<hash>, move the old reference to
candidate, and update reference to the new build.

Options:
  --cleanup   Remove old unreferenced enyo_<hash> binaries after promotion.
  --dry-run   Print what would happen without building or changing files.
  --keep N    With --cleanup, keep the newest N unreferenced engines.
  -h, --help  Show this help text.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cleanup)
            cleanup_old=true
            ;;
        --dry-run)
            dry_run=true
            ;;
        --keep)
            if [[ $# -lt 2 || ! "$2" =~ ^[0-9]+$ ]]; then
                echo "ERROR: --keep needs a non-negative integer" >&2
                exit 1
            fi
            keep_count="$2"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

if [[ ! "$keep_count" =~ ^[0-9]+$ ]]; then
    echo "ERROR: KEEP_ENGINES needs a non-negative integer" >&2
    exit 1
fi

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

target_path() {
    local target="$1"
    if [[ "$target" == /* ]]; then
        printf '%s\n' "$(cd "$(dirname "$target")" && pwd -P)/$(basename "$target")"
    else
        printf '%s\n' "$assets_dir/$target"
    fi
}

cleanup_old_engines() {
    local reference_target candidate_target target keep_path engine keep
    local keep_paths=()
    local old_engines=()
    local remove=()
    local unreferenced_seen=0

    reference_target="$(readlink "$assets_dir/reference" || true)"
    candidate_target="$(readlink "$assets_dir/candidate" || true)"

    for target in "$reference_target" "$candidate_target"; do
        if [[ -n "$target" ]]; then
            keep_paths+=("$(target_path "$target")")
        fi
    done

    shopt -s nullglob
    local engine_candidates=("$assets_dir"/enyo_[0-9a-f]*)
    shopt -u nullglob
    if [[ ${#engine_candidates[@]} -gt 0 ]]; then
        while IFS= read -r engine; do
            if [[ -f "$engine" && ! -L "$engine" ]]; then
                old_engines+=("$engine")
            fi
        done < <(ls -1t "${engine_candidates[@]}")
    fi

    for engine in "${old_engines[@]}"; do
        keep=false
        for keep_path in "${keep_paths[@]}"; do
            if [[ "$engine" == "$keep_path" ]]; then
                keep=true
                break
            fi
        done

        if [[ "$keep" == true ]]; then
            continue
        fi

        if (( unreferenced_seen < keep_count )); then
            ((unreferenced_seen += 1))
            continue
        fi

        remove+=("$engine")
    done

    if [[ ${#remove[@]} -eq 0 ]]; then
        echo "cleanup: nothing to remove"
        return 0
    fi

    for engine in "${remove[@]}"; do
        if [[ "$dry_run" == true ]]; then
            echo "cleanup: would remove $engine"
        else
            echo "cleanup: removing $engine"
            rm -f -- "$engine"
        fi
    done
}

if [[ "$dry_run" == true && "$cleanup_old" == true ]]; then
    cleanup_old_engines
    exit 0
fi

main_commit="$(git -C "$repo_root" rev-parse --verify "$main_ref^{commit}")"
short_hash="$(git -C "$repo_root" rev-parse --short "$main_commit")"
dest="$assets_dir/enyo_$short_hash"

if [[ "$dry_run" == true ]]; then
    echo "dry-run: would build $main_ref ($short_hash) as $dest"
    echo "dry-run: would set candidate -> $old_reference"
    echo "dry-run: would set reference -> enyo_$short_hash"
    exit 0
fi

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

if [[ "$cleanup_old" != true ]]; then
    exit 0
fi

cleanup_old_engines
