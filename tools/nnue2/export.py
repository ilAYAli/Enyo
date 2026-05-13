from __future__ import annotations

import argparse

import torch

from model import EnyoNNUE2, export_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="PyTorch state_dict .pt")
    ap.add_argument("--out", required=True, help="Output Berserk-format .nn")
    args = ap.parse_args()

    model = EnyoNNUE2()
    model.load_state_dict(torch.load(args.state, map_location="cpu"))
    export_model(model, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
