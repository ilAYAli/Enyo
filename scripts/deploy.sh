#!/usr/bin/env bash
set -euo pipefail

# Determine paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINES_DIR="$HOME/assets/engines"

cd "$REPO_ROOT"

usage() {
    echo "Usage: $0 [--candidate] [--reference]"
}

# Argument parsing
MODES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --candidate)
            MODES+=("candidate")
            ;;
        --reference)
            MODES+=("reference")
            ;;
        *) usage; exit 1 ;;
    esac
    shift
done

if [[ ${#MODES[@]} -eq 0 ]]; then
    usage
    exit 1
fi

# Git checks
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Error: Working directory is dirty. Please commit or stash changes."
    exit 1
fi

HASH=$(git rev-parse --short HEAD)
echo "Target git hash: $HASH"

# Build verification
BUILD_NEEDED=true
if [[ -f "build/enyo" ]]; then
    # Verify if the existing binary matches the current hash
    # Example output: id Enyo Release v.263e103 built 31/05-26
    if ./build/enyo <<< "quit" | grep -q "$HASH"; then
        echo "Existing build matches current hash ($HASH). Skipping build."
        BUILD_NEEDED=false
    else
        echo "Existing build does not match $HASH."
    fi
fi

if [[ "$BUILD_NEEDED" == "true" ]]; then
    echo "Building Enyo..."
    # Follow user specified build steps
    rm -rf build/*
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
fi

# Deploy
mkdir -p "$ENGINES_DIR"
DEST_EXE="enyo_$HASH"

echo "Copying to $ENGINES_DIR/$DEST_EXE..."
cp build/enyo "$ENGINES_DIR/$DEST_EXE"

for MODE in "${MODES[@]}"; do
    echo "Updating $MODE symlink..."
    ln -sfn "$ENGINES_DIR/$DEST_EXE" "$ENGINES_DIR/$MODE"
    ls -l "$ENGINES_DIR/$MODE"
done