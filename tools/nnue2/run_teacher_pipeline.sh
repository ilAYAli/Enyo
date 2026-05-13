#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$(git rev-parse --show-toplevel)}
python=${PYTHON:-/home/petter/.venv/bin/python}
engine=${ENGINE:-stockfish}
source_jsonl=${SOURCE_JSONL:-/home/petter/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl}
init_net=${INIT_NET:-/home/petter/code/cpp/chess/enyo/nnue/berserk-d43206fe90e4.nn}
out=${OUT:-/home/petter/tmp/enyo_teacher/sf_d12_150k_$(date +%Y%m%d_%H%M%S)}
sample_rows=${SAMPLE_ROWS:-150000}
depth=${DEPTH:-12}
shards=${SHARDS:-8}
threads=${THREADS:-1}
hash_mb=${HASH:-128}
train_epochs=${TRAIN_EPOCHS:-4}
batch_size=${BATCH_SIZE:-8192}
lr=${LR:-2e-7}
val_rows=${VAL_ROWS:-10000}

mkdir -p "$out/shards"
ln -sfn "$out" "$(dirname "$out")/latest"
cd "$repo"

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

sample=$out/sample.jsonl
labeled=$out/labeled.jsonl
packed=$out/labeled_packed
candidate=$out/huber_lr${lr}_e${train_epochs}

log "sample rows=$sample_rows"
"$python" tools/nnue2/sample_jsonl.py \
  --input "$source_jsonl" \
  --output "$sample" \
  --rows "$sample_rows" \
  --seed 20260510 \
  --unique-fen | tee "$out/sample.log"

log "label with $engine depth=$depth shards=$shards"
pids=()
cleanup_labelers() {
  if ((${#pids[@]})); then
    kill "${pids[@]}" >/dev/null 2>&1 || true
  fi
}
trap cleanup_labelers INT TERM EXIT
for shard in $(seq 0 $((shards - 1))); do
  shard_out="$out/shards/label_${shard}.jsonl"
  "$python" tools/nnue2/label_with_uci.py \
    --input "$sample" \
    --output "$shard_out" \
    --engine "$engine" \
    --depth "$depth" \
    --threads "$threads" \
    --hash "$hash_mb" \
    --shard-count "$shards" \
    --shard-index "$shard" \
    --max-abs-cp 1600 \
    --progress 1000 \
    > "$out/shards/label_${shard}.log" 2>&1 &
  pids+=("$!")
done

label_status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    label_status=1
  fi
done
trap - INT TERM EXIT
if ((label_status != 0)); then
  log "teacher labeling failed; see $out/shards/*.log"
  exit "$label_status"
fi

log "merge labels"
cat "$out"/shards/label_*.jsonl > "$labeled"
wc -l "$labeled" | tee "$out/labeled.wc"
labeled_rows=$(wc -l < "$labeled" | tr -d ' ')
val_skip=0
if ((labeled_rows > val_rows)); then
  val_skip=$((labeled_rows - val_rows))
fi

log "pack labeled rows"
"$python" tools/nnue2/pack_dataset.py \
  --input "$labeled" \
  --out-dir "$packed" \
  --progress 25000 | tee "$out/pack.log"

log "baseline metrics"
"$python" tools/nnue2/eval_dataset.py \
  --net "$init_net" \
  --data "$packed" \
  --skip "$val_skip" \
  --rows "$val_rows" \
  --batch-size "$batch_size" \
  --target-clamp 1600 \
  --buckets | tee "$out/baseline_metrics.log"

log "train candidate"
mkdir -p "$candidate"
"$python" tools/nnue2/train.py \
  --data "$packed" \
  --init-from-nn "$init_net" \
  --out "$candidate/model.pt" \
  --out-nn "$candidate/model.nn" \
  --objective huber \
  --huber-beta 200 \
  --select-metric mae \
  --target-clamp 1600 \
  --epochs "$train_epochs" \
  --lr "$lr" \
  --weight-decay 1e-6 \
  --batch-size "$batch_size" \
  --val-rows "$val_rows" \
  --workers 0 \
  --trainable all \
  --device cuda | tee "$candidate/train.log"

log "candidate metrics"
"$python" tools/nnue2/eval_dataset.py \
  --net "$candidate/model.nn" \
  --data "$packed" \
  --skip "$val_skip" \
  --rows "$val_rows" \
  --batch-size "$batch_size" \
  --target-clamp 1600 \
  --buckets | tee "$candidate/metrics.log"

log "done out=$out candidate=$candidate/model.nn"
