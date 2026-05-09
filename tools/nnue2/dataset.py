"""Self-play JSONL dataset for the NNUE2 trainer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

import enyo_nnue2 as nn2


class FenScoreDataset(Dataset):
    def __init__(self, rows: Iterable[dict]):
        self.items: list[
            tuple[list[int], list[int], int, float, float, float]
        ] = []
        for row in rows:
            fen = row["fen"]
            pieces, stm = nn2.parse_fen(fen)
            pieces.sort(key=lambda item: item[2])
            w_feats = nn2.features_from_pieces(pieces, nn2.WHITE)
            b_feats = nn2.features_from_pieces(pieces, nn2.BLACK)
            phase_scale = nn2.phase_scale_from_pieces(pieces)
            self.items.append((
                w_feats,
                b_feats,
                stm,
                float(row["score"]),
                float(row.get("wdl", 0.5)),
                phase_scale,
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
        w, b, stm, score, wdl, phase_scale = self.items[idx]
        return (
            torch.tensor(w, dtype=torch.long),
            torch.tensor(b, dtype=torch.long),
            torch.tensor(stm, dtype=torch.long),
            torch.tensor(score, dtype=torch.float32),
            torch.tensor(wdl, dtype=torch.float32),
            torch.tensor(phase_scale, dtype=torch.float32),
        )


def collate(batch):
    w_all, b_all = [], []
    w_offsets, b_offsets = [0], [0]
    stms, scores, wdls, phase_scales = [], [], [], []
    for w, b, stm, score, wdl, phase_scale in batch:
        w_all.append(w)
        b_all.append(b)
        w_offsets.append(w_offsets[-1] + len(w))
        b_offsets.append(b_offsets[-1] + len(b))
        stms.append(stm)
        scores.append(score)
        wdls.append(wdl)
        phase_scales.append(phase_scale)

    return (
        torch.cat(w_all) if w_all else torch.empty(0, dtype=torch.long),
        torch.cat(b_all) if b_all else torch.empty(0, dtype=torch.long),
        torch.tensor(w_offsets[:-1], dtype=torch.long),
        torch.tensor(b_offsets[:-1], dtype=torch.long),
        torch.stack(stms),
        torch.stack(scores),
        torch.stack(wdls),
        torch.stack(phase_scales),
    )
