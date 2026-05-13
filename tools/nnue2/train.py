"""Train/fine-tune Enyo's 1024-hidden Berserk-format NNUE2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import count_rows, load_score_dataset
from model import EnyoNNUE2, export_model, load_model_from_nn


MPE_SCALE = 2.5 / 400.0
MPE_EXPONENT = 2.5


def mpe25_loss(pred_cp: torch.Tensor, target_cp: torch.Tensor,
               wdl: torch.Tensor, wdl_lambda: float) -> torch.Tensor:
    pred_p = torch.sigmoid(pred_cp * MPE_SCALE)
    target_p = torch.sigmoid(target_cp * MPE_SCALE)
    if wdl_lambda < 1.0:
        target_p = wdl_lambda * target_p + (1.0 - wdl_lambda) * wdl
    return ((pred_p - target_p).abs() ** MPE_EXPONENT).mean()


def load_source_map(path: str | Path) -> dict[str, int]:
    p = Path(path)
    source_map_path = p / "source_map.json"
    if not source_map_path.exists():
        return {}
    raw = json.loads(source_map_path.read_text())
    return {str(name): int(source_id) for name, source_id in raw.items()}


def parse_source_values(items: list[str] | None, *, value_name: str,
                        source_map: dict[str, int]
                        ) -> dict[int, float]:
    values: dict[int, float] = {}
    for item in items or []:
        source, value = item.split("=", 1)
        try:
            source_id = int(source)
        except ValueError as exc:
            if source not in source_map:
                known = ", ".join(sorted(source_map)) or "<none>"
                raise SystemExit(
                    f"unknown {value_name} source '{source}'. "
                    f"Known sources: {known}"
                ) from exc
            source_id = source_map[source]
        values[source_id] = float(value)
    return values


def source_value_tensor(source_ids: torch.Tensor, default: float,
                        overrides: dict[int, float]) -> torch.Tensor:
    values = torch.full_like(source_ids, float(default), dtype=torch.float32)
    for source_id, value in overrides.items():
        values = torch.where(
            source_ids == source_id,
            torch.full_like(values, float(value)),
            values)
    return values


def score_loss(pred: torch.Tensor, target: torch.Tensor,
               wdl: torch.Tensor, source_ids: torch.Tensor,
               args: argparse.Namespace) -> torch.Tensor:
    if args.objective == "mpe25":
        pred_p = torch.sigmoid(pred * MPE_SCALE)
        target_p = torch.sigmoid(target * MPE_SCALE)
        lambdas = source_value_tensor(
            source_ids, args.wdl_lambda, args.source_wdl_lambdas)
        target_p = lambdas * target_p + (1.0 - lambdas) * wdl
        losses = (pred_p - target_p).abs() ** MPE_EXPONENT
    elif args.objective == "huber":
        losses = F.smooth_l1_loss(
            pred, target, beta=args.huber_beta, reduction="none")
    elif args.objective == "mse":
        losses = (pred - target) ** 2
    else:
        raise ValueError(f"unknown objective: {args.objective}")

    if args.source_loss_weights:
        weights = source_value_tensor(
            source_ids, 1.0, args.source_loss_weights)
        return (losses * weights).sum() / weights.sum().clamp_min(1.0)
    return losses.mean()


def selection_value(metrics: dict[str, float], args: argparse.Namespace) -> float:
    value = metrics[args.select_metric]
    return -value if args.select_metric == "sign" else value


@torch.no_grad()
def eval_metrics(model: EnyoNNUE2, loader: DataLoader, args: argparse.Namespace
                 ) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    sign_sum = 0
    sign_n = 0
    n = 0
    for w, b, w_off, b_off, stm, y, wdl, phase_scale, source_ids in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        wdl = wdl.to(args.device)
        phase_scale = phase_scale.to(args.device)
        source_ids = source_ids.to(args.device)
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)
        pred = model(w, b, w_off, b_off, stm, phase_scale)
        loss = score_loss(pred, y, wdl, source_ids, args)
        err = pred - y
        sign_mask = y != 0
        batch_n = len(y)
        loss_sum += float(loss) * batch_n
        mae_sum += float(err.abs().sum())
        mse_sum += float((err * err).sum())
        sign_sum += int(((pred[sign_mask] > 0) == (y[sign_mask] > 0)).sum())
        sign_n += int(sign_mask.sum())
        n += batch_n
    model.train()
    denom = max(1, n)
    return {
        "loss": loss_sum / denom,
        "mse": mse_sum / denom,
        "mae": mae_sum / denom,
        "sign": sign_sum / max(1, sign_n),
    }


def train(args: argparse.Namespace) -> EnyoNNUE2:
    print(f"loading train rows from {args.data}", flush=True)
    train_limit = args.max_rows
    val_skip = args.skip_rows + (args.max_rows if args.max_rows > 0 else 0)
    if not args.val and args.max_rows == 0 and args.val_rows > 0:
        total_rows = count_rows(args.data)
        train_limit = max(0, total_rows - args.skip_rows - args.val_rows)
        val_skip = args.skip_rows + train_limit
        print(
            f"reserving final {args.val_rows} rows for validation "
            f"(total={total_rows}, train_limit={train_limit}, "
            f"val_skip={val_skip})", flush=True)

    train_set, collate_fn = load_score_dataset(
        args.data, limit=train_limit, skip=args.skip_rows)
    print(f"train rows: {len(train_set)}", flush=True)
    val_set = None
    if args.val:
        print(f"loading val rows from {args.val}", flush=True)
        val_set, val_collate_fn = load_score_dataset(
            args.val, limit=args.val_rows)
    elif args.val_rows > 0:
        print(f"loading val rows from {args.data} at skip={val_skip}",
              flush=True)
        val_set, val_collate_fn = load_score_dataset(
            args.data, limit=args.val_rows, skip=val_skip)
    if val_set is not None:
        print(f"val rows: {len(val_set)}", flush=True)

    if args.init_from_nn:
        print(f"initializing from {args.init_from_nn}", flush=True)
        model = load_model_from_nn(args.init_from_nn, device=args.device)
    else:
        model = EnyoNNUE2(init=args.init).to(args.device)

    if args.trainable != "all":
        for param in model.parameters():
            param.requires_grad_(False)
        if args.trainable in ("float-head", "output"):
            for param in model.output.parameters():
                param.requires_grad_(True)
        if args.trainable == "float-head":
            for param in model.l2.parameters():
                param.requires_grad_(True)
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"trainable={args.trainable} params={trainable_params}",
              flush=True)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"))
    val_loader = (DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=val_collate_fn, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"))
        if val_set is not None else None)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    best_metric = float("inf")
    best_display = float("inf")
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        mae_sum = 0.0
        mse_sum = 0.0
        n = 0
        for w, b, w_off, b_off, stm, y, wdl, phase_scale, source_ids in train_loader:
            w = w.to(args.device)
            b = b.to(args.device)
            w_off = w_off.to(args.device)
            b_off = b_off.to(args.device)
            stm = stm.to(args.device)
            y = y.to(args.device)
            wdl = wdl.to(args.device)
            phase_scale = phase_scale.to(args.device)
            source_ids = source_ids.to(args.device)
            if args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)

            pred = model(w, b, w_off, b_off, stm, phase_scale)
            loss = score_loss(pred, y, wdl, source_ids, args)

            opt.zero_grad()
            loss.backward()
            opt.step()

            err = (pred.detach() - y)
            mae_sum += float(err.abs().sum())
            mse_sum += float((err * err).sum())
            n += len(y)

        line = (f"epoch {epoch:4d} train mse={mse_sum / max(1, n):10.2f} "
                f"mae={mae_sum / max(1, n):7.2f}")
        val_metrics = None
        if val_loader is not None:
            val_metrics = eval_metrics(model, val_loader, args)
            line += (
                f" val loss={val_metrics['loss']:.6f}"
                f" mse={val_metrics['mse']:10.2f}"
                f" mae={val_metrics['mae']:7.2f}"
                f" sign={val_metrics['sign'] * 100:5.2f}%")
            metric = selection_value(val_metrics, args)
            if metric < best_metric:
                best_metric = metric
                best_display = val_metrics[args.select_metric]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if args.patience > 0:
                line += (f" best_{args.select_metric}="
                         f"{best_display:.6f} bad={bad}")
        print(line, flush=True)

        if args.patience > 0 and val_metrics is not None and bad >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True, help="Output .pt state_dict")
    ap.add_argument("--out-nn", default=None, help="Optional exported .nn")
    ap.add_argument("--init-from-nn", default=None,
                    help="Start from an existing Berserk-format .nn")
    ap.add_argument("--init", default="kaiming",
                    choices=["kaiming", "berserk-ish"])
    ap.add_argument("--objective", default="mpe25",
                    choices=["mse", "huber", "mpe25"])
    ap.add_argument("--huber-beta", type=float, default=200.0,
                    help="SmoothL1/Huber transition in centipawns.")
    ap.add_argument("--select-metric", default="loss",
                    choices=["loss", "mse", "mae", "sign"],
                    help="Validation metric used to keep the best checkpoint. "
                         "sign is maximized; the others are minimized.")
    ap.add_argument("--wdl-lambda", type=float, default=0.75)
    ap.add_argument("--source-wdl-lambda", action="append", default=[],
                    metavar="SOURCE_ID=VALUE",
                    help="Override MPE WDL lambda for one packed source id. "
                         "Use 1.0 for CP-only sources such as eval DB rows.")
    ap.add_argument("--source-loss-weight", action="append", default=[],
                    metavar="SOURCE_ID=VALUE",
                    help="Per-source loss weight for source-aware packed "
                         "datasets.")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-6)
    ap.add_argument("--target-clamp", type=float, default=0.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--skip-rows", type=int, default=0)
    ap.add_argument("--val-rows", type=int, default=0)
    ap.add_argument("--trainable", default="all",
                    choices=["all", "float-head", "output"],
                    help="'all' trains every weight. 'float-head' freezes "
                         "the quantized input/L1 layers and trains only "
                         "L2+output floats. 'output' trains only the final "
                         "linear layer.")
    args = ap.parse_args()
    source_map = load_source_map(args.data)
    args.source_wdl_lambdas = parse_source_values(
        args.source_wdl_lambda, value_name="source-wdl-lambda",
        source_map=source_map)
    args.source_loss_weights = parse_source_values(
        args.source_loss_weight, value_name="source-loss-weight",
        source_map=source_map)
    if source_map:
        print(f"source_map={source_map}", flush=True)
    if args.source_wdl_lambdas:
        print(f"source_wdl_lambdas={args.source_wdl_lambdas}", flush=True)
    if args.source_loss_weights:
        print(f"source_loss_weights={args.source_loss_weights}", flush=True)

    model = train(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.cpu().state_dict(), out)
    print(f"wrote {out}")

    if args.out_nn:
        export_model(model, args.out_nn)
        print(f"wrote {args.out_nn}")


if __name__ == "__main__":
    main()
