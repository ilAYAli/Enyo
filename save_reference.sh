#!/bin/bash
# Save current build/enyo as a reference engine

if [ ! -f build/enyo ]; then
    echo "Error: build/enyo not found"
    exit 1
fi

# Get git hash
git_hash=$(git log --oneline -1 --format="%h" 2>/dev/null || echo "unknown")

mkdir -p ../assets/engines
dest="../assets/engines/enyo_${git_hash}"

cp build/enyo "$dest"
chmod +x "$dest"

echo "✓ Saved reference: $dest"
./build/enyo 2>&1 | head -1
