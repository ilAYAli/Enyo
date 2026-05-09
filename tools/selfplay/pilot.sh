#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
workspace_root="$(cd "$repo_root/.." && pwd)"
assets_dir="$workspace_root/assets"
if [[ ! -d "$assets_dir" && -d "$HOME/code/cpp/chess/assets" ]]; then
    assets_dir="$HOME/code/cpp/chess/assets"
fi

runner=""
engine="$repo_root/build/enyo"
nnue2_file="$repo_root/nnue/berserk-d43206fe90e4.nn"
book="$assets_dir/books/UHO_Lichess_4852_v1.epd"
games=1000
shard_games=1000
concurrency=8
threads=4
depth=8
tc=""
out_dir="$HOME/tmp/enyo_selfplay/pilot_$(date +%Y%m%d_%H%M%S)"
device="cuda"
epochs=5
batch_size=4096
max_rows=0
val_rows=5000
max_abs_cp=2000
trainable="float-head"
lr="1e-5"
wdl_lambda="0.75"
python_bin="${PYTHON:-}"
if [[ -z "$python_bin" ]]; then
    if [[ -x "$HOME/.venv/bin/python" ]]; then
        python_bin="$HOME/.venv/bin/python"
    else
        python_bin="python3"
    fi
fi

usage() {
    cat <<EOF
Usage: $0 [options]

Run a small Enyo-owned NNUE2 data/training pilot:
  1. fastchess Enyo-vs-Enyo PGN with eval comments
  2. PGN -> JSONL training rows
  3. dataset stats + baseline eval
  4. fine-tune/export a Berserk-format .nn candidate

Options:
  --runner PATH
  --engine PATH
  --nnue2-file PATH
  --book PATH
  --games N
  --shard-games N
  --concurrency N
  --threads N
  --depth N
  --tc TC
  --out-dir PATH
  --device DEVICE
  --epochs N
  --batch-size N
  --max-rows N
  --val-rows N
  --max-abs-cp N
  --trainable all|float-head|output
  --lr FLOAT
  --wdl-lambda FLOAT
  --python PATH
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner) runner="$2"; shift 2 ;;
        --engine) engine="$2"; shift 2 ;;
        --nnue2-file) nnue2_file="$2"; shift 2 ;;
        --book) book="$2"; shift 2 ;;
        --games) games="$2"; shift 2 ;;
        --shard-games) shard_games="$2"; shift 2 ;;
        --concurrency) concurrency="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --depth) depth="$2"; shift 2 ;;
        --tc) tc="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --epochs) epochs="$2"; shift 2 ;;
        --batch-size) batch_size="$2"; shift 2 ;;
        --max-rows) max_rows="$2"; shift 2 ;;
        --val-rows) val_rows="$2"; shift 2 ;;
        --max-abs-cp) max_abs_cp="$2"; shift 2 ;;
        --trainable) trainable="$2"; shift 2 ;;
        --lr) lr="$2"; shift 2 ;;
        --wdl-lambda) wdl_lambda="$2"; shift 2 ;;
        --python) python_bin="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

abs_path() {
"$python_bin" - "$1" <<'PY'
import os
import sys
print(os.path.abspath(os.path.expanduser(sys.argv[1])))
PY
}

out_dir="$(abs_path "$out_dir")"
engine="$(abs_path "$engine")"
nnue2_file="$(abs_path "$nnue2_file")"
book="$(abs_path "$book")"
mkdir -p "$out_dir"

pgn="$out_dir/selfplay.pgn"
jsonl="$out_dir/selfplay.jsonl"
stats="$out_dir/selfplay.stats.json"
state="$out_dir/own_net_pilot.pt"
net="$out_dir/own_net_pilot.nn"

echo "pilot: out_dir=$out_dir"

selfplay_args=(
    --engine "$engine"
    --nnue2-file "$nnue2_file"
    --book "$book"
    --games "$games"
    --shard-games "$shard_games"
    --concurrency "$concurrency"
    --threads "$threads"
    --depth "$depth"
    --output "$pgn"
)
if [[ -n "$runner" ]]; then
    selfplay_args=(--runner "$runner" "${selfplay_args[@]}")
fi
if [[ -n "$tc" ]]; then
    selfplay_args+=(--tc "$tc")
fi

"$script_dir/run_selfplay.sh" "${selfplay_args[@]}"

"$python_bin" "$script_dir/pgn_to_jsonl.py" "$pgn" \
    --output "$jsonl" \
    --stats "$stats" \
    --skip-plies 8 \
    --min-depth 1 \
    --max-abs-cp "$max_abs_cp"

wc -l "$jsonl"
cat "$stats"

min_rows=$((games * 20))
if ((min_rows < 1000)); then
    min_rows=1000
fi
audit_depth=0
if [[ -z "$tc" ]]; then
    audit_depth="$depth"
fi
"$python_bin" "$script_dir/audit_jsonl.py" "$jsonl" \
    --min-rows "$min_rows" \
    --expected-depth "$audit_depth" \
    --cap-cp 2045 \
    --max-capped-pct 0.5

"$python_bin" "$repo_root/tools/nnue2/roundtrip.py" "$nnue2_file"
"$python_bin" "$repo_root/tools/nnue2/eval_dataset.py" \
    --net "$nnue2_file" \
    --data "$jsonl" \
    --rows 50000 \
    --batch-size "$batch_size" \
    --device "$device"

"$python_bin" "$repo_root/tools/nnue2/train.py" \
    --data "$jsonl" \
    --init-from-nn "$nnue2_file" \
    --max-rows "$max_rows" \
    --val-rows "$val_rows" \
    --device "$device" \
    --epochs "$epochs" \
    --batch-size "$batch_size" \
    --lr "$lr" \
    --objective mpe25 \
    --wdl-lambda "$wdl_lambda" \
    --trainable "$trainable" \
    --out "$state" \
    --out-nn "$net"

"$python_bin" "$repo_root/tools/nnue2/eval_dataset.py" \
    --net "$net" \
    --data "$jsonl" \
    --rows 50000 \
    --batch-size "$batch_size" \
    --device "$device"

echo "pilot: wrote $net"
