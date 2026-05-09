from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from dataset import load_score_dataset
from model import load_model_from_nn
from train import MPE_EXPONENT, MPE_SCALE


BUCKETS = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 300.0),
    (300.0, 800.0),
    (800.0, 1600.0),
    (1600.0, float("inf")),
)


def bucket_name(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:.0f}"
    return f"{lo:.0f}-{hi:.0f}"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--target-clamp", type=float, default=0.0)
    ap.add_argument("--buckets", action="store_true",
                    help="Print metrics grouped by absolute target score.")
    args = ap.parse_args()

    ds, collate_fn = load_score_dataset(
        args.data, limit=args.rows, skip=args.skip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    model = load_model_from_nn(args.net, device=args.device)
    model.eval()

    n = 0
    mae = 0.0
    mse = 0.0
    mpe = 0.0
    sign = 0
    sign_n = 0
    pred_sum = 0.0
    target_sum = 0.0
    pred_sq_sum = 0.0
    target_sq_sum = 0.0
    pred_target_sum = 0.0
    bucket_stats = [
        {"n": 0, "mae": 0.0, "sign": 0, "sign_n": 0,
         "pred": 0.0, "target": 0.0}
        for _ in BUCKETS
    ]
    for w, b, w_off, b_off, stm, y, _wdl, phase_scale in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        phase_scale = phase_scale.to(args.device)
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)
        pred = model(w, b, w_off, b_off, stm, phase_scale)
        err = pred - y
        sign_mask = y != 0
        mae += float(err.abs().sum())
        mse += float((err * err).sum())
        mpe += float(
            ((torch.sigmoid(pred * MPE_SCALE)
              - torch.sigmoid(y * MPE_SCALE)).abs() ** MPE_EXPONENT).sum())
        sign += int(((pred[sign_mask] > 0) == (y[sign_mask] > 0)).sum())
        sign_n += int(sign_mask.sum())
        pred_sum += float(pred.sum())
        target_sum += float(y.sum())
        pred_sq_sum += float((pred * pred).sum())
        target_sq_sum += float((y * y).sum())
        pred_target_sum += float((pred * y).sum())
        batch_n = len(y)
        n += batch_n

        if args.buckets:
            abs_y = y.abs()
            abs_err = err.abs()
            for idx, (lo, hi) in enumerate(BUCKETS):
                mask = (abs_y >= lo) if math.isinf(hi) else (
                    (abs_y >= lo) & (abs_y < hi))
                count = int(mask.sum())
                if count == 0:
                    continue
                b = bucket_stats[idx]
                b["n"] += count
                b["mae"] += float(abs_err[mask].sum())
                b["pred"] += float(pred[mask].sum())
                b["target"] += float(y[mask].sum())
                bucket_sign = mask & sign_mask
                b["sign"] += int(
                    ((pred[bucket_sign] > 0)
                     == (y[bucket_sign] > 0)).sum())
                b["sign_n"] += int(bucket_sign.sum())

    print(f"rows={n}")
    print(f"mae={mae / max(1, n):.3f}")
    print(f"mse={mse / max(1, n):.3f}")
    print(f"rmse={math.sqrt(mse / max(1, n)):.3f}")
    print(f"mpe25={mpe / max(1, n):.8f}")
    print(f"sign={sign / max(1, sign_n) * 100:.2f}%")
    print(f"wrong_sign={sign_n - sign}/{sign_n}")
    pred_mean = pred_sum / max(1, n)
    target_mean = target_sum / max(1, n)
    cov = pred_target_sum / max(1, n) - pred_mean * target_mean
    pred_var = pred_sq_sum / max(1, n) - pred_mean * pred_mean
    target_var = target_sq_sum / max(1, n) - target_mean * target_mean
    corr = cov / math.sqrt(max(1e-12, pred_var * target_var))
    slope = cov / max(1e-12, target_var)
    print(f"bias={pred_mean - target_mean:.3f}")
    print(f"pred_mean={pred_mean:.3f}")
    print(f"target_mean={target_mean:.3f}")
    print(f"corr={corr:.6f}")
    print(f"slope={slope:.6f}")

    if args.buckets:
        print("bucket       rows     mae    sign   pred_mean target_mean")
        for (lo, hi), b in zip(BUCKETS, bucket_stats):
            rows = b["n"]
            if rows == 0:
                continue
            sign_pct = b["sign"] / max(1, b["sign_n"]) * 100.0
            print(
                f"{bucket_name(lo, hi):>9} {rows:8d}"
                f" {b['mae'] / rows:7.2f}"
                f" {sign_pct:7.2f}%"
                f" {b['pred'] / rows:10.2f}"
                f" {b['target'] / rows:11.2f}")


if __name__ == "__main__":
    main()
