#!/usr/bin/env bash
set -euo pipefail

source ~/.ntfy 2>/dev/null || true

notify() {
  local msg="$1"
  echo "$(date --iso-8601=seconds) $msg"
  "$HOME/scripts/notifai.sh" "$msg" codex_1 >/dev/null 2>&1 || true
  if [ -n "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" ]; then
    curl -fsS -m 10 -u "${NTFY_AUTH:-${LICHESS_NTFY_AUTH:-}}" \
      -d "$msg" https://ntfy.wahlman.no/ping >/dev/null 2>&1 || true
  else
    curl -fsS -m 10 -d "$msg" https://ntfy.wahlman.no/ping \
      >/dev/null 2>&1 || true
  fi
}

PY="$HOME/.venv/bin/python"
REPO="$HOME/code/cpp/chess/enyo"
TOOLS="$REPO/tools/nnue2"
RUN="$HOME/tmp/enyo_teacher/source_mix_1m_$(date +%Y%m%d_%H%M%S)"
INIT="$REPO/nnue/berserk-d43206fe90e4.nn"
ENGINE="$HOME/code/cpp/chess/assets/engines/enyo_206059e"
BUGS="$REPO/bugs"
SPRT="$HOME/code/cpp/chess/sprt/sprt"
BOOK="$HOME/code/cpp/chess/assets/books/UHO_Lichess_4852_v1.epd"

D16_JSON="$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/labeled.jsonl"
BINPACK_JSON="$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/binpack.jsonl"
D16_PACKED="$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/packed"
SELFPLAY_PACKED="$HOME/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled_packed"
LICHESS_PACKED="$HOME/tmp/enyo_teacher/controlled_30m_20260512_105006/val/lichess_tail100k"
BINPACK_PACKED="$HOME/tmp/enyo_teacher/binpack_test79_cp1600_5m_20260512/packed"
BASE_SUMMARY="$HOME/tmp/enyo_teacher/sf_d16_bucket1m_20260512_225554/baseline/replay_gates_force/summary.txt"

mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1
cd "$REPO"

notify "Enyo NNUE source-mix 1M: start $RUN"

for required in \
  "$PY" "$INIT" "$ENGINE" "$SPRT" "$D16_JSON" "$BINPACK_JSON" \
  "$D16_PACKED" "$SELFPLAY_PACKED" "$LICHESS_PACKED" "$BINPACK_PACKED" \
  "$BASE_SUMMARY"
do
  if [ ! -e "$required" ]; then
    notify "Enyo NNUE source-mix 1M: missing $required"
    exit 1
  fi
done

issue_lines() {
  local file="$1"
  rg --no-config \
    "^(inaccuracy|mistake|blunder|timeout|game:|replay exit|missing replay log)" \
    "$file" 2>/dev/null || true
}

sample_and_pack_binpack() {
  local dir="$RUN/binpack1m_all_huber_lr1e6_e8"
  mkdir -p "$dir"
  if [ ! -s "$dir/source.jsonl" ]; then
    notify "Enyo NNUE source-mix 1M: sample binpack-only 1M"
    "$PY" "$TOOLS/sample_jsonl.py" \
      --input "$BINPACK_JSON" \
      --output "$dir/source.jsonl" \
      --rows 1000000 \
      --seed 2026051301
  fi
  if [ ! -s "$dir/packed/meta.json" ]; then
    notify "Enyo NNUE source-mix 1M: pack binpack-only 1M"
    "$PY" "$TOOLS/pack_dataset.py" \
      --input "$dir/source.jsonl" \
      --out-dir "$dir/packed" \
      --progress 200000
  fi
}

mix_and_pack() {
  local dir="$RUN/mix_d16_500k_binpack_500k_all_huber_lr1e6_e8"
  mkdir -p "$dir"
  if [ ! -s "$dir/source.jsonl" ]; then
    notify "Enyo NNUE source-mix 1M: mix d16 500k + binpack 500k"
    "$PY" "$TOOLS/mix_jsonl.py" \
      --output "$dir/source.jsonl" \
      --source "$D16_JSON:500000" \
      --source "$BINPACK_JSON:500000" \
      --seed 2026051302 \
      --progress 200000
  fi
  if [ ! -s "$dir/packed/meta.json" ]; then
    notify "Enyo NNUE source-mix 1M: pack mixed 1M"
    "$PY" "$TOOLS/pack_dataset.py" \
      --input "$dir/source.jsonl" \
      --out-dir "$dir/packed" \
      --progress 200000
  fi
}

train_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  if [ ! -s "$dir/model.nn" ]; then
    notify "Enyo NNUE source-mix 1M: train $tag"
    "$PY" "$TOOLS/train.py" \
      --data "$dir/packed" \
      --init-from-nn "$INIT" \
      --objective huber \
      --huber-beta 200 \
      --select-metric mae \
      --epochs 8 \
      --patience 2 \
      --batch-size 8192 \
      --lr 1e-6 \
      --weight-decay 1e-6 \
      --target-clamp 1600 \
      --device cuda \
      --workers 2 \
      --val-rows 50000 \
      --trainable all \
      --out "$dir/model.pt" \
      --out-nn "$dir/model.nn" | tee "$dir/train.log"
  fi
}

eval_one() {
  local net="$1"
  local data="$2"
  local skip="$3"
  local out="$4"
  "$PY" "$TOOLS/eval_dataset.py" \
    --net "$net" \
    --data "$data" \
    --skip "$skip" \
    --rows 50000 \
    --batch-size 8192 \
    --device cuda \
    --target-clamp 1600 \
    --buckets > "$out"
}

eval_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  mkdir -p "$dir/eval"
  notify "Enyo NNUE source-mix 1M: eval $tag"
  eval_one "$INIT" "$D16_PACKED" 938632 "$dir/eval/d16_baseline.txt"
  eval_one "$dir/model.nn" "$D16_PACKED" 938632 "$dir/eval/d16_candidate.txt"
  eval_one "$INIT" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay_baseline.txt"
  eval_one "$dir/model.nn" "$SELFPLAY_PACKED" 20839426 "$dir/eval/selfplay_candidate.txt"
  eval_one "$INIT" "$LICHESS_PACKED" 0 "$dir/eval/lichess_baseline.txt"
  eval_one "$dir/model.nn" "$LICHESS_PACKED" 0 "$dir/eval/lichess_candidate.txt"
  eval_one "$INIT" "$BINPACK_PACKED" 4900000 "$dir/eval/binpack_baseline.txt"
  eval_one "$dir/model.nn" "$BINPACK_PACKED" 4900000 "$dir/eval/binpack_candidate.txt"

  "$PY" - "$dir/eval" <<'PY' | tee "$dir/static_summary.txt"
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
labels = ["d16", "selfplay", "lichess", "binpack"]


def metric(path: pathlib.Path, key: str) -> float:
    text = path.read_text()
    m = re.search(rf"^{key}=([-+0-9.]+)%?$", text, re.M)
    if not m:
        raise SystemExit(f"missing {key} in {path}")
    return float(m.group(1))


score = 0.0
print("dataset       base_mae cand_mae delta_mae base_sign cand_sign delta_sign")
for label in labels:
    b = root / f"{label}_baseline.txt"
    c = root / f"{label}_candidate.txt"
    b_mae = metric(b, "mae")
    c_mae = metric(c, "mae")
    b_sign = metric(b, "sign")
    c_sign = metric(c, "sign")
    d_mae = b_mae - c_mae
    d_sign = c_sign - b_sign
    score += d_mae
    print(
        f"{label:10} {b_mae:8.3f} {c_mae:8.3f} {d_mae:9.3f}"
        f" {b_sign:9.2f} {c_sign:9.2f} {d_sign:10.2f}"
    )
print(f"score={score:.3f}")
PY
}

make_wrapper() {
  local tag="$1"
  local dir="$RUN/$tag"
  local gate="$dir/replay_gates"
  mkdir -p "$gate/home/.config/enyo"
  cat > "$gate/home/.config/enyo/settings.json" <<JSON
{
  "constants": {
    "threads": 4,
    "Hash": 1024,
    "nnue2_file": "$dir/model.nn",
    "logfile": "$gate/enyo.log"
  },
  "uci_options": {
    "Threads": 4,
    "Hash": 1024,
    "nnue2_file": "$dir/model.nn"
  }
}
JSON
  cat > "$gate/enyo_candidate.sh" <<WRAP
#!/usr/bin/env bash
export HOME="$gate/home"
exec "$ENGINE" "\$@"
WRAP
  chmod +x "$gate/enyo_candidate.sh"
}

gate_candidate() {
  local tag="$1"
  local dir="$RUN/$tag"
  local gate="$dir/replay_gates"
  make_wrapper "$tag"
  notify "Enyo NNUE source-mix 1M: replay gate $tag"
  : > "$gate/replay.log"
  : > "$gate/summary.txt"
  for f in \
    "EnyoBot vs Lynx_BOT - jjThVRPN.log" \
    "Hypersion vs EnyoBot - npmgxvIO.log" \
    "JustinBot15 vs EnyoBot - JZaA98Uv.log" \
    "EnyoBot vs stage270 - 2DRMYfOm_oot.log" \
    "stage270 vs EnyoBot - kp3inZBb.log"
  do
    printf "== %s ==\n" "$f" | tee -a "$gate/replay.log" "$gate/summary.txt"
    set +e
    /home/petter/local/bin/replay --engine "$gate/enyo_candidate.sh" \
      "$BUGS/$f" 2>&1 \
      | tee -a "$gate/replay.log" \
      | awk '/=== Summary ===/ { seen=1 } seen' \
      | tee -a "$gate/summary.txt"
    replay_rc=${PIPESTATUS[0]}
    set -e
    if [ "$replay_rc" -ne 0 ]; then
      echo "replay exit $replay_rc: $f" \
        | tee -a "$gate/replay.log" "$gate/summary.txt"
    fi
    printf "\n" | tee -a "$gate/replay.log" "$gate/summary.txt"
  done

  issue_lines "$BASE_SUMMARY" | sed -E "s/[[:space:]]+/ /g" | sort -u \
    > "$RUN/baseline.issues"
  issue_lines "$gate/summary.txt" | sed -E "s/[[:space:]]+/ /g" | sort -u \
    > "$dir/issues"
  comm -13 "$RUN/baseline.issues" "$dir/issues" > "$dir/new_issues"
  rg --no-config "^(mistake|blunder|timeout|game:|replay exit|missing replay log)" \
    "$dir/new_issues" > "$dir/new_serious_issues" || true

  local new_issues new_serious score
  new_issues=$(wc -l < "$dir/new_issues" | tr -d " ")
  new_serious=$(wc -l < "$dir/new_serious_issues" | tr -d " ")
  score=$(awk -F= '/^score=/ {print $2}' "$dir/static_summary.txt" | tail -1)
  {
    echo "candidate=$dir/model.nn"
    echo "static_score=$score"
    echo "new_issues=$new_issues"
    echo "new_serious_issues=$new_serious"
    echo
    echo "new issues:"
    cat "$dir/new_issues"
  } | tee "$dir/gate_decision.txt"

  if [ "$new_serious" -eq 0 ]; then
    printf "%s\t%s\t%s\n" "$score" "$tag" "$dir/model.nn" \
      >> "$RUN/passed_candidates.tsv"
  fi
}

run_sprt_for_best() {
  if [ ! -s "$RUN/passed_candidates.tsv" ]; then
    notify "Enyo NNUE source-mix 1M: no candidate passed replay gate"
    exit 0
  fi
  local best_line score tag net
  best_line=$(sort -k1,1nr "$RUN/passed_candidates.tsv" | head -1)
  score=$(printf "%s" "$best_line" | cut -f1)
  tag=$(printf "%s" "$best_line" | cut -f2)
  net=$(printf "%s" "$best_line" | cut -f3)
  if ! "$PY" - "$score" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0.0 else 1)
PY
  then
    notify \
      "Enyo NNUE source-mix 1M: best replay-clean candidate static_score=$score; no SPRT"
    exit 0
  fi
  notify "Enyo NNUE source-mix 1M: SPRT start $tag static_score=$score"
  mkdir -p "$RUN/sprt"
  exec "$SPRT" \
    --candidate "$ENGINE" \
    --reference "$ENGINE" \
    --candidate-option "nnue2_file=$net" \
    --candidate-option "Hash=1024" \
    --reference-option "nnue2_file=$INIT" \
    --reference-option "Hash=1024" \
    --book "$BOOK" \
    --games 2000 \
    --concurrency 6 \
    --threads 4 \
    --tc "10+0.1" \
    --log-dir "$RUN/sprt" \
    --name "${tag}_vs_reference_net" \
    --ntfy-url "https://ntfy.wahlman.no/sprt"
}

: > "$RUN/passed_candidates.tsv"

sample_and_pack_binpack
train_candidate "binpack1m_all_huber_lr1e6_e8"
eval_candidate "binpack1m_all_huber_lr1e6_e8"
gate_candidate "binpack1m_all_huber_lr1e6_e8"

mix_and_pack
train_candidate "mix_d16_500k_binpack_500k_all_huber_lr1e6_e8"
eval_candidate "mix_d16_500k_binpack_500k_all_huber_lr1e6_e8"
gate_candidate "mix_d16_500k_binpack_500k_all_huber_lr1e6_e8"

run_sprt_for_best
