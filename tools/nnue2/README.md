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
