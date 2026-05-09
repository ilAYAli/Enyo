"""Train/fine-tune Enyo's 1024-hidden Berserk-format NNUE2."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import FenScoreDataset, collate
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


@torch.no_grad()
def eval_metrics(model: EnyoNNUE2, loader: DataLoader, args: argparse.Namespace
                 ) -> tuple[float, float, float]:
    model.eval()
    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    for w, b, w_off, b_off, stm, y, wdl, phase_scale in loader:
        w = w.to(args.device)
        b = b.to(args.device)
        w_off = w_off.to(args.device)
        b_off = b_off.to(args.device)
        stm = stm.to(args.device)
        y = y.to(args.device)
        wdl = wdl.to(args.device)
        phase_scale = phase_scale.to(args.device)
        if args.target_clamp > 0:
            y = torch.clamp(y, -args.target_clamp, args.target_clamp)
        pred = model(w, b, w_off, b_off, stm, phase_scale)
        if args.objective == "mpe25":
            loss = mpe25_loss(pred, y, wdl, args.wdl_lambda)
        else:
            loss = ((pred - y) ** 2).mean()
        err = pred - y
        batch_n = len(y)
        loss_sum += float(loss) * batch_n
        mae_sum += float(err.abs().sum())
        mse_sum += float((err * err).sum())
        n += batch_n
    model.train()
    denom = max(1, n)
    return loss_sum / denom, mse_sum / denom, mae_sum / denom


def train(args: argparse.Namespace) -> EnyoNNUE2:
    print(f"loading train rows from {args.data}")
    train_set = FenScoreDataset.from_jsonl(
        args.data, limit=args.max_rows, skip=args.skip_rows)
    print(f"train rows: {len(train_set)}")
    val_set = None
    if args.val:
        print(f"loading val rows from {args.val}")
        val_set = FenScoreDataset.from_jsonl(args.val, limit=args.val_rows)
    elif args.val_rows > 0:
        val_skip = args.skip_rows + (args.max_rows if args.max_rows > 0 else 0)
        print(f"loading val rows from {args.data} at skip={val_skip}")
        val_set = FenScoreDataset.from_jsonl(
            args.data, limit=args.val_rows, skip=val_skip)
    if val_set is not None:
        print(f"val rows: {len(val_set)}")

    if args.init_from_nn:
        print(f"initializing from {args.init_from_nn}")
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
        print(f"trainable={args.trainable} params={trainable_params}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"))
    val_loader = (DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.workers,
        pin_memory=args.device.startswith("cuda"))
        if val_set is not None else None)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr, weight_decay=args.weight_decay)

    best_loss = float("inf")
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        mae_sum = 0.0
        mse_sum = 0.0
        n = 0
        for w, b, w_off, b_off, stm, y, wdl, phase_scale in train_loader:
            w = w.to(args.device)
            b = b.to(args.device)
            w_off = w_off.to(args.device)
            b_off = b_off.to(args.device)
            stm = stm.to(args.device)
            y = y.to(args.device)
            wdl = wdl.to(args.device)
            phase_scale = phase_scale.to(args.device)
            if args.target_clamp > 0:
                y = torch.clamp(y, -args.target_clamp, args.target_clamp)

            pred = model(w, b, w_off, b_off, stm, phase_scale)
            if args.objective == "mpe25":
                loss = mpe25_loss(pred, y, wdl, args.wdl_lambda)
            else:
                loss = ((pred - y) ** 2).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            err = (pred.detach() - y)
            mae_sum += float(err.abs().sum())
            mse_sum += float((err * err).sum())
            n += len(y)

        line = (f"epoch {epoch:4d} train mse={mse_sum / max(1, n):10.2f} "
                f"mae={mae_sum / max(1, n):7.2f}")
        val_loss = None
        if val_loader is not None:
            val_loss, val_mse, val_mae = eval_metrics(model, val_loader, args)
            line += (f" val loss={val_loss:.6f} mse={val_mse:10.2f} "
                     f"mae={val_mae:7.2f}")
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }
                bad = 0
            else:
                bad += 1
            if args.patience > 0:
                line += f" best_loss={best_loss:.6f} bad={bad}"
        print(line, flush=True)

        if args.patience > 0 and val_loss is not None and bad >= args.patience:
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
                    choices=["mse", "mpe25"])
    ap.add_argument("--wdl-lambda", type=float, default=0.75)
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
