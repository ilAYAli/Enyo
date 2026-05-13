# Enyo NNUE2 Trainer

This trains or fine-tunes Enyo's Berserk-format `1024` hidden-neuron
network and exports a `.nn` file loadable through:

```text
setoption name nnue2_file value nnue/<file>.nn
```

Typical first run is fine-tuning the current Berserk net on Enyo
self-play rows:

```bash
python3 tools/nnue2/train.py \
  --data ~/tmp/enyo_selfplay/50k_d8_sharded_20260506_113050/selfplay_d8_50k.jsonl \
  --init-from-nn nnue/berserk-d43206fe90e4.nn \
  --max-rows 1000000 \
  --val-rows 50000 \
  --device cuda \
  --epochs 20 \
  --batch-size 4096 \
  --lr 1e-5 \
  --objective mpe25 \
  --wdl-lambda 0.75 \
  --out ~/tmp/nnue2_selfplay.pt \
  --out-nn ~/tmp/nnue2_selfplay.nn
```

Roundtrip check for loader/exporter correctness:

```bash
python3 tools/nnue2/roundtrip.py nnue/berserk-d43206fe90e4.nn
```

Target/data quality check before training:

```bash
python3 tools/nnue2/audit_targets.py \
  ~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl \
  --rows 100000
```

Import filtered Lichess eval DB rows:

```bash
python3 tools/nnue2/import_lichess_eval.py \
  --input ~/data/lichess/lichess_db_eval.jsonl.zst \
  --output ~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl \
  --rows 5000000 \
  --min-depth 18 \
  --max-abs-cp 1600 \
  --unique-fen
```

The importer converts Lichess eval DB centipawns from white POV to Enyo's
side-to-move training convention, and rejects non-standard material positions.

Import official Stockfish `.binpack` rows:

```bash
c++ -O3 -std=c++23 \
  -I ~/tmp/nnue-pytorch/data_loader/cpp/lib \
  tools/nnue2/binpack_to_jsonl.cpp \
  -lfmt \
  -o ~/tmp/binpack_to_jsonl

~/tmp/binpack_to_jsonl \
  --input ~/code/cpp/chess/assets/test79-may2022-16tb7p-filter-v6-dd.min-mar2023.unmin.high-simple-eval-1k.min-v2.binpack \
  --output ~/tmp/enyo_teacher/binpack_test79/binpack.jsonl \
  --limit 5000000 \
  --max-abs-cp 1600
```

The converter emits Enyo JSONL rows, converts Stockfish's internal eval unit to
centipawns with `100 * score / 208`, and rejects illegal/extreme positions.

Mix Enyo self-play labels with imported Lichess eval rows:

```bash
python3 tools/nnue2/mix_jsonl.py \
  --output ~/tmp/enyo_teacher/mixed_20m_selfplay_5m_lichess.jsonl \
  --seed 20260511 \
  --source ~/tmp/enyo_teacher/sf_d12_20m_20260510_115338/labeled.jsonl:20000000 \
  --source ~/tmp/enyo_teacher/lichess_eval_d18_standard/lichess_eval.jsonl:5000000
```

The mixer streams selected rows and intentionally does not do cross-source
FEN dedupe. Dedupe each input before mixing if exact uniqueness is required.

Notes:

- The model includes Enyo's `ScaleEval` phase scaling during training,
  because search uses the scaled value.
- `--init-from-nn` is the sane default. Training this architecture from
  scratch needs much more data and time.
- The current dataset loader is intentionally simple and loads selected
  JSONL rows into memory. Use `--max-rows` for JSONL pilots.
- For multi-million-row runs, first pack the JSONL to mmap arrays:

```bash
python3 tools/nnue2/pack_dataset.py \
  --input ~/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay.jsonl \
  --out-dir ~/tmp/enyo_selfplay/d8_6m_20260509_165354/selfplay_packed
```

Then pass the packed directory to `--data`.
