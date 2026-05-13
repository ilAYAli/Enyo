"""Self-play JSONL dataset for the NNUE2 trainer."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

import enyo_nnue2 as nn2


class FenScoreDataset(Dataset):
    def __init__(self, rows: Iterable[dict]):
        self.items: list[
            tuple[list[int], list[int], int, float, float, float, int]
        ] = []
        source_map: dict[str, int] = {}
        for row in rows:
            fen = row["fen"]
            pieces, stm = nn2.parse_fen(fen)
            pieces.sort(key=lambda item: item[2])
            w_feats = nn2.features_from_pieces(pieces, nn2.WHITE)
            b_feats = nn2.features_from_pieces(pieces, nn2.BLACK)
            phase_scale = nn2.phase_scale_from_pieces(pieces)
            source_name = str(
                row.get("source_type") or row.get("teacher") or "unknown")
            if source_name not in source_map:
                source_map[source_name] = len(source_map)
            self.items.append((
                w_feats,
                b_feats,
                stm,
                float(row["score"]),
                float(row.get("wdl", 0.5)),
                phase_scale,
                source_map[source_name],
            ))

    @classmethod
    def from_jsonl(cls, path: str | Path, *,
                   limit: int = 0, skip: int = 0) -> "FenScoreDataset":
        rows = []
        with Path(path).open() as f:
            for i, line in enumerate(f):
                if i < skip:
                    continue
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if limit > 0 and len(rows) >= limit:
                    break
        return cls(rows)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        w, b, stm, score, wdl, phase_scale, source_id = self.items[idx]
        return (
            torch.tensor(w, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(wdl, dtype=torch.float32),
            torch.tensor(phase_scale, dtype=torch.float32),
            torch.tensor(source_id, dtype=torch.long),
        )


class PackedFenScoreDataset(Dataset):
    def __init__(self, path: str | Path, *,
                 limit: int = 0, skip: int = 0) -> None:
        root = Path(path)
        self.w_features = np.load(root / "white_features.npy", mmap_mode="r")
        self.b_features = np.load(root / "black_features.npy", mmap_mode="r")
        self.counts = np.load(root / "counts.npy", mmap_mode="r")
        self.stms = np.load(root / "stm.npy", mmap_mode="r")
        self.scores = np.load(root / "score.npy", mmap_mode="r")
        self.wdls = np.load(root / "wdl.npy", mmap_mode="r")
        self.phase_scales = np.load(root / "phase_scale.npy", mmap_mode="r")
        source_id_path = root / "source_id.npy"
        self.source_ids = (
            np.load(source_id_path, mmap_mode="r")
            if source_id_path.exists() else None)

        rows = len(self.counts)
        self.start = min(skip, rows)
        self.end = rows if limit <= 0 else min(rows, self.start + limit)

    def __len__(self) -> int:
        return self.end - self.start

    def __getitem__(self, idx: int):
        real_idx = self.start + idx
        return (
            self.w_features[real_idx],
            self.b_features[real_idx],
            self.counts[real_idx],
            self.stms[real_idx],
            self.scores[real_idx],
            self.wdls[real_idx],
            self.phase_scales[real_idx],
            0 if self.source_ids is None else self.source_ids[real_idx],
        )


def collate(batch):
    w_all, b_all = [], []
    w_offsets, b_offsets = [0], [0]
    stms, scores, wdls, phase_scales, source_ids = [], [], [], [], []
    for w, b, stm, score, wdl, phase_scale, source_id in batch:
        w_all.append(w)
        b_all.append(b)
        w_offsets.append(w_offsets[-1] + len(w))
        b_offsets.append(b_offsets[-1] + len(b))
        stms.append(stm)
        scores.append(score)
        wdls.append(wdl)
        phase_scales.append(phase_scale)
        source_ids.append(source_id)

    return (
        torch.cat(w_all) if w_all else torch.empty(0, dtype=torch.long),
        torch.cat(b_all) if b_all else torch.empty(0, dtype=torch.long),
        torch.tensor(w_offsets[:-1], dtype=torch.long),
        torch.tensor(b_offsets[:-1], dtype=torch.long),
        torch.stack(stms),
        torch.stack(scores),
        torch.stack(wdls),
        torch.stack(phase_scales),
        torch.stack(source_ids),
    )


def collate_packed(batch):
    w_rows, b_rows, counts, stms, scores, wdls, phase_scales, source_ids = zip(
        *batch)
    w_padded = torch.as_tensor(np.stack(w_rows), dtype=torch.long)
    b_padded = torch.as_tensor(np.stack(b_rows), dtype=torch.long)
    counts_t = torch.as_tensor(np.asarray(counts), dtype=torch.long)
    max_features = w_padded.shape[1]
    mask = (torch.arange(max_features).unsqueeze(0)
            < counts_t.unsqueeze(1))
    offsets = torch.cat((
        torch.zeros(1, dtype=torch.long),
        torch.cumsum(counts_t, dim=0)[:-1],
    ))
    return (
        w_padded[mask],
        b_padded[mask],
        offsets,
        offsets.clone(),
        torch.as_tensor(np.asarray(stms), dtype=torch.long),
        torch.as_tensor(np.asarray(scores), dtype=torch.float32),
        torch.as_tensor(np.asarray(wdls), dtype=torch.float32),
        torch.as_tensor(np.asarray(phase_scales), dtype=torch.float32),
        torch.as_tensor(np.asarray(source_ids), dtype=torch.long),
    )


def count_rows(path: str | Path) -> int:
    p = Path(path)
    if p.is_dir():
        counts = np.load(p / "counts.npy", mmap_mode="r")
        return len(counts)

    n = 0
    with p.open() as handle:
        for line in handle:
            if line.strip():
                n += 1
    return n


def load_score_dataset(path: str | Path, *, limit: int = 0, skip: int = 0
                       ) -> tuple[Dataset, Callable]:
    p = Path(path)
    if p.is_dir():
        return PackedFenScoreDataset(p, limit=limit, skip=skip), collate_packed
    return FenScoreDataset.from_jsonl(p, limit=limit, skip=skip), collate
