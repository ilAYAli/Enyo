#!/bin/bash
set -e

echo "=== SPRT Test Setup ==="
echo "Reference: enyo_dcdd1fe (commit dcdd1fe)"
echo "Candidate: search/protect-quiet-checks-v1"
echo ""

cd ~/code/cpp/chess/enyo

# Pull latest changes
echo "Fetching latest changes..."
git fetch origin

# Build reference engine from dcdd1fe
echo ""
echo "Building reference engine (dcdd1fe)..."
git checkout dcdd1fe
cd build
make clean
make -j16
cd ..
mkdir -p ../assets/engines
cp build/enyo ../assets/engines/enyo_dcdd1fe
echo "Reference built: ../assets/engines/enyo_dcdd1fe"

# Switch to candidate branch and build
echo ""
echo "Building candidate (search/protect-quiet-checks-v1)..."
git checkout search/protect-quiet-checks-v1
git pull origin search/protect-quiet-checks-v1
cd build
make clean
make -j16
cd ..
echo "Candidate built: ./build/enyo"

# Verify the fix works
echo ""
echo "Verifying fix on problematic FEN..."
printf 'uci\nsetoption name Threads value 1\nisready\nucinewgame\nposition fen 1r6/2R2p1k/p2p1P2/2pb2BP/3b1NP1/5P2/6K1/8 b - - 2 32\ngo depth 12\nquit\n' | ./build/enyo | grep "^bestmove"

# Run SPRT
echo ""
echo "Starting SPRT (1000 games, 8 concurrent)..."
../assets/scripts/sprt --games 1000 --concurrent 8 --reference ../assets/engines/enyo_dcdd1fe --candidate ./build/enyo --ntfy-url https://ntfy.wahlman.no/sprt
