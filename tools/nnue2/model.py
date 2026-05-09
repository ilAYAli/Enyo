"""PyTorch model matching Enyo NNUE2's Berserk-format architecture."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn_pt

import enyo_nnue2 as nn2


class EnyoNNUE2(nn_pt.Module):
    def __init__(self, init: str = "kaiming"):
        super().__init__()
        self.embed = nn_pt.EmbeddingBag(
            nn2.N_FEATURES, nn2.N_HIDDEN, mode="sum")
        self.input_bias = nn_pt.Parameter(torch.zeros(nn2.N_HIDDEN))
        self.l1_weight = nn_pt.Parameter(torch.zeros(nn2.N_L2, nn2.N_L1))
        self.l1_bias = nn_pt.Parameter(torch.zeros(nn2.N_L2))
        self.l2 = nn_pt.Linear(nn2.N_L2, nn2.N_L3)
        self.output = nn_pt.Linear(nn2.N_L3, nn2.N_OUTPUT)

        if init == "kaiming":
            nn_pt.init.normal_(self.embed.weight, std=math.sqrt(2.0 / 32.0))
            nn_pt.init.normal_(self.l1_weight, std=math.sqrt(2.0 / nn2.N_L1))
            nn_pt.init.normal_(self.l2.weight, std=math.sqrt(2.0 / nn2.N_L2))
            nn_pt.init.normal_(self.output.weight, std=math.sqrt(2.0 / nn2.N_L3))
        elif init == "berserk-ish":
            nn_pt.init.normal_(self.embed.weight, std=64.0)
            nn_pt.init.normal_(self.l1_weight, std=4.0)
            nn_pt.init.normal_(self.l2.weight, std=0.02)
            nn_pt.init.normal_(self.output.weight, std=0.02)
        else:
            raise ValueError(f"unknown init '{init}'")

    def accumulator(self, feats: torch.Tensor, offsets: torch.Tensor
                    ) -> torch.Tensor:
        return self.embed(feats, offsets) + self.input_bias

    def raw_forward(self, w_feats: torch.Tensor, b_feats: torch.Tensor,
                    w_offsets: torch.Tensor, b_offsets: torch.Tensor,
                    stm: torch.Tensor) -> torch.Tensor:
        w_acc = self.accumulator(w_feats, w_offsets)
        b_acc = self.accumulator(b_feats, b_offsets)

        stm_f = stm.unsqueeze(-1).float()
        us = (1.0 - stm_f) * w_acc + stm_f * b_acc
        them = stm_f * w_acc + (1.0 - stm_f) * b_acc
        acc = torch.cat([us, them], dim=-1)

        x0 = torch.clamp(acc, min=0.0, max=float(127 << nn2.QUANT1_BITS))
        x0 = x0 / float(1 << nn2.QUANT1_BITS)
        x1 = torch.relu(x0 @ self.l1_weight.t() + self.l1_bias)
        x2 = torch.relu(self.l2(x1))
        return self.output(x2).squeeze(-1) / nn2.EVAL_DIVISOR

    def forward(self, w_feats: torch.Tensor, b_feats: torch.Tensor,
                w_offsets: torch.Tensor, b_offsets: torch.Tensor,
                stm: torch.Tensor, phase_scale: torch.Tensor
                ) -> torch.Tensor:
        raw = self.raw_forward(w_feats, b_feats, w_offsets, b_offsets, stm)
        return torch.clamp(raw * phase_scale, min=-2045.0, max=2045.0)


def load_model_from_nn(path: str | Path, device: str = "cpu") -> EnyoNNUE2:
    net = nn2.load_net(path)
    model = EnyoNNUE2(init="kaiming")
    with torch.no_grad():
        model.embed.weight.copy_(torch.from_numpy(net.input_weights.astype(np.float32)))
        model.input_bias.copy_(torch.from_numpy(net.input_biases.astype(np.float32)))
        model.l1_weight.copy_(torch.from_numpy(net.l1_weights.astype(np.float32)))
        model.l1_bias.copy_(torch.from_numpy(net.l1_biases.astype(np.float32)))
        model.l2.weight.copy_(torch.from_numpy(net.l2_weights.astype(np.float32)))
        model.l2.bias.copy_(torch.from_numpy(net.l2_biases.astype(np.float32)))
        model.output.weight.copy_(
            torch.from_numpy(net.output_weights.astype(np.float32)).view(1, -1))
        model.output.bias.copy_(torch.tensor([net.output_bias], dtype=torch.float32))
    return model.to(device)


def _clip_round(arr: np.ndarray, lo: int, hi: int, name: str) -> np.ndarray:
    rounded = np.rint(arr)
    clipped = np.clip(rounded, lo, hi)
    n_clip = int(np.sum(rounded != clipped))
    if n_clip:
        print(f"warning: {name}: clipped {n_clip} values")
    return clipped


def export_model(model: EnyoNNUE2, path: str | Path) -> None:
    m = model.cpu()
    iw = _clip_round(
        m.embed.weight.detach().numpy(),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
        "input_weights").astype(np.int16)
    ib = _clip_round(
        m.input_bias.detach().numpy(),
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
        "input_biases").astype(np.int16)
    l1w = _clip_round(
        m.l1_weight.detach().numpy(),
        np.iinfo(np.int8).min,
        np.iinfo(np.int8).max,
        "l1_weights").astype(np.int8)
    l1b = _clip_round(
        m.l1_bias.detach().numpy(),
        np.iinfo(np.int32).min,
        np.iinfo(np.int32).max,
        "l1_biases").astype(np.int32)
    net = nn2.Net(
        input_weights=iw,
        input_biases=ib,
        l1_weights=l1w,
        l1_biases=l1b,
        l2_weights=m.l2.weight.detach().numpy().astype(np.float32),
        l2_biases=m.l2.bias.detach().numpy().astype(np.float32),
        output_weights=m.output.weight.detach().numpy().reshape(-1).astype(np.float32),
        output_bias=float(m.output.bias.detach().numpy().item()),
    )
    nn2.write_net(net, path)
