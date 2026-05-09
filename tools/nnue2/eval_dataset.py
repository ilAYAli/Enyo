from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from dataset import FenScoreDataset, collate
from model import load_model_from_nn
from train import MPE_EXPONENT, MPE_SCALE


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ds = FenScoreDataset.from_jsonl(args.data, limit=args.rows, skip=args.skip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate)
    model = load_model_from_nn(args.net, device=args.device)
    model.eval()

    n = 0
    mae = 0.0
    mse = 0.0
    mpe = 0.0
    sign = 0
    for w, b, w_off, b_off, stm, y, _wdl, phase_scale in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        phase_scale = phase_scale.to(args.device)
        pred = model(w, b, w_off, b_off, stm, phase_scale)
        err = pred - y
        mae += float(err.abs().sum())
        mse += float((err * err).sum())
        mpe += float(
            ((torch.sigmoid(pred * MPE_SCALE)
              - torch.sigmoid(y * MPE_SCALE)).abs() ** MPE_EXPONENT).sum())
        sign += int(((pred > 0) == (y > 0)).sum())
        n += len(y)

    print(f"rows={n}")
    print(f"mae={mae / max(1, n):.3f}")
    print(f"mse={mse / max(1, n):.3f}")
    print(f"mpe25={mpe / max(1, n):.8f}")
    print(f"sign={sign / max(1, n) * 100:.2f}%")


if __name__ == "__main__":
    main()
