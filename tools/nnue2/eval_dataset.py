from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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


def empty_stats() -> dict[str, float]:
    return {
        "n": 0,
        "mae": 0.0,
        "mse": 0.0,
        "mpe": 0.0,
        "sign": 0,
        "sign_n": 0,
        "pred_sum": 0.0,
        "target_sum": 0.0,
        "pred_sq_sum": 0.0,
        "target_sq_sum": 0.0,
        "pred_target_sum": 0.0,
    }


def update_stats(stats: dict[str, float], pred: torch.Tensor,
                 target: torch.Tensor) -> None:
    err = pred - target
    sign_mask = target != 0
    stats["n"] += len(target)
    stats["mae"] += float(err.abs().sum())
    stats["mse"] += float((err * err).sum())
    stats["mpe"] += float(
        ((torch.sigmoid(pred * MPE_SCALE)
          - torch.sigmoid(target * MPE_SCALE)).abs() ** MPE_EXPONENT).sum())
    stats["sign"] += int(
        ((pred[sign_mask] > 0) == (target[sign_mask] > 0)).sum())
    stats["sign_n"] += int(sign_mask.sum())
    stats["pred_sum"] += float(pred.sum())
    stats["target_sum"] += float(target.sum())
    stats["pred_sq_sum"] += float((pred * pred).sum())
    stats["target_sq_sum"] += float((target * target).sum())
    stats["pred_target_sum"] += float((pred * target).sum())


def derived_stats(stats: dict[str, float]) -> dict[str, float]:
    n = max(1.0, stats["n"])
    pred_mean = stats["pred_sum"] / n
    target_mean = stats["target_sum"] / n
    cov = stats["pred_target_sum"] / n - pred_mean * target_mean
    pred_var = stats["pred_sq_sum"] / n - pred_mean * pred_mean
    target_var = stats["target_sq_sum"] / n - target_mean * target_mean
    return {
        "rows": stats["n"],
        "mae": stats["mae"] / n,
        "mse": stats["mse"] / n,
        "rmse": math.sqrt(stats["mse"] / n),
        "mpe25": stats["mpe"] / n,
        "sign": stats["sign"] / max(1.0, stats["sign_n"]),
        "wrong_sign": stats["sign_n"] - stats["sign"],
        "sign_n": stats["sign_n"],
        "bias": pred_mean - target_mean,
        "pred_mean": pred_mean,
        "target_mean": target_mean,
        "corr": cov / math.sqrt(max(1e-12, pred_var * target_var)),
        "slope": cov / max(1e-12, target_var),
    }


def source_names(data_path: str) -> dict[int, str]:
    source_map_path = Path(data_path) / "source_map.json"
    if not source_map_path.exists():
        return {}
    raw = json.loads(source_map_path.read_text())
    return {int(source_id): str(name) for name, source_id in raw.items()}


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
    ap.add_argument("--sources", action="store_true",
                    help="Print metrics grouped by source id/name.")
    args = ap.parse_args()

    ds, collate_fn = load_score_dataset(
        args.data, limit=args.rows, skip=args.skip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn)
    model = load_model_from_nn(args.net, device=args.device)
    model.eval()

    overall = empty_stats()
    by_source: dict[int, dict[str, float]] = {}
    names_by_source = source_names(args.data)
    bucket_stats = [
        {"n": 0, "mae": 0.0, "sign": 0, "sign_n": 0,
         "pred": 0.0, "target": 0.0}
        for _ in BUCKETS
    ]
    for w, b, w_off, b_off, stm, y, _wdl, phase_scale, source_ids in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        phase_scale = phase_scale.to(args.device)
        source_ids = source_ids.to(args.device)
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)
        pred = model(w, b, w_off, b_off, stm, phase_scale)
        err = pred - y
        sign_mask = y != 0
        update_stats(overall, pred, y)

        if args.sources:
            for source_id_tensor in torch.unique(source_ids):
                source_id = int(source_id_tensor.item())
                mask = source_ids == source_id_tensor
                stats = by_source.setdefault(source_id, empty_stats())
                update_stats(stats, pred[mask], y[mask])

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

    final = derived_stats(overall)
    print(f"rows={int(final['rows'])}")
    print(f"mae={final['mae']:.3f}")
    print(f"mse={final['mse']:.3f}")
    print(f"rmse={final['rmse']:.3f}")
    print(f"mpe25={final['mpe25']:.8f}")
    print(f"sign={final['sign'] * 100:.2f}%")
    print(f"wrong_sign={int(final['wrong_sign'])}/{int(final['sign_n'])}")
    print(f"bias={final['bias']:.3f}")
    print(f"pred_mean={final['pred_mean']:.3f}")
    print(f"target_mean={final['target_mean']:.3f}")
    print(f"corr={final['corr']:.6f}")
    print(f"slope={final['slope']:.6f}")

    if args.sources:
        print("source                 rows     mae    sign    bias    corr   slope")
        for source_id, stats in sorted(by_source.items()):
            item = derived_stats(stats)
            name = names_by_source.get(source_id, str(source_id))
            print(
                f"{name[:18]:18} {int(item['rows']):8d}"
                f" {item['mae']:7.2f}"
                f" {item['sign'] * 100:7.2f}%"
                f" {item['bias']:7.2f}"
                f" {item['corr']:7.3f}"
                f" {item['slope']:7.3f}")

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
