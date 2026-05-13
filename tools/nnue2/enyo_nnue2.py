"""Python reference helpers for Enyo's Berserk-format NNUE2.

The constants and feature-index formula mirror src/nnue2.hpp.  The file
format is Berserk v13's .nn layout as loaded by NNUE2::LoadNetwork().
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 1, 2, 3, 4, 5, 6

N_KING_BUCKETS = 16
N_PIECE_TYPES = 12
N_SQUARES = 64
N_FEATURES = N_KING_BUCKETS * N_PIECE_TYPES * N_SQUARES
N_HIDDEN = 1024
N_L1 = 2 * N_HIDDEN
N_L2 = 16
N_L3 = 32
N_OUTPUT = 1

QUANT1_BITS = 5
EVAL_DIVISOR = 32.0

NETWORK_SIZE = (
    N_FEATURES * N_HIDDEN * np.dtype(np.int16).itemsize
    + N_HIDDEN * np.dtype(np.int16).itemsize
    + N_L1 * N_L2 * np.dtype(np.int8).itemsize
    + N_L2 * np.dtype(np.int32).itemsize
    + N_L2 * N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * np.dtype(np.float32).itemsize
    + N_L3 * N_OUTPUT * np.dtype(np.float32).itemsize
    + N_OUTPUT * np.dtype(np.float32).itemsize
)

KING_BUCKETS: tuple[int, ...] = (
    15, 15, 14, 14, 14, 14, 15, 15,
    15, 15, 14, 14, 14, 14, 15, 15,
    13, 13, 12, 12, 12, 12, 13, 13,
    13, 13, 12, 12, 12, 12, 13, 13,
    11, 10,  9,  8,  8,  9, 10, 11,
    11, 10,  9,  8,  8,  9, 10, 11,
     7,  6,  5,  4,  4,  5,  6,  7,
     3,  2,  1,  0,  0,  1,  2,  3,
)

_FEN_PIECE = {
    "p": PAWN,
    "n": KNIGHT,
    "b": BISHOP,
    "r": ROOK,
    "q": QUEEN,
    "k": KING,
}


@dataclass
class Net:
    input_weights: np.ndarray   # (N_FEATURES, N_HIDDEN) int16
    input_biases: np.ndarray    # (N_HIDDEN,) int16
    l1_weights: np.ndarray      # (N_L2, N_L1) int8
    l1_biases: np.ndarray       # (N_L2,) int32
    l2_weights: np.ndarray      # (N_L3, N_L2) float32
    l2_biases: np.ndarray       # (N_L3,) float32
    output_weights: np.ndarray  # (N_L3,) float32
    output_bias: float


def to_berserk_sq(enyo_sq: int) -> int:
    return enyo_sq ^ 63


def feature_index(piece_type: int, piece_color: int, enyo_sq: int,
                  enyo_kingsq: int, view: int) -> int:
    if piece_type == 0:
        return 0

    piece = ((piece_type - 1) << 1) | piece_color
    sq = to_berserk_sq(enyo_sq)
    kingsq = to_berserk_sq(enyo_kingsq)

    op = 6 * ((piece ^ view) & 0x1) + (piece >> 1)
    ok = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ kingsq
    osq = (7 * (0 if (kingsq & 4) else 1)) ^ (56 * view) ^ sq

    return KING_BUCKETS[ok] * 12 * 64 + op * 64 + osq


def parse_fen(fen: str) -> tuple[list[tuple[int, int, int]], int]:
    parts = fen.split()
    board_part, stm_part = parts[0], parts[1]
    pieces: list[tuple[int, int, int]] = []
    rank = 7
    file_idx = 0
    for ch in board_part:
        if ch == "/":
            rank -= 1
            file_idx = 0
            continue
        if ch.isdigit():
            file_idx += int(ch)
            continue

        # Enyo square convention: h1=0, g1=1, ..., a8=63.
        sq = rank * 8 + (7 - file_idx)
        color = WHITE if ch.isupper() else BLACK
        pieces.append((_FEN_PIECE[ch.lower()], color, sq))
        file_idx += 1

    return pieces, (WHITE if stm_part == "w" else BLACK)


def features_from_pieces(pieces: Sequence[tuple[int, int, int]],
                         view: int) -> list[int]:
    king_sq = next(sq for pt, color, sq in pieces
                   if pt == KING and color == view)
    return [feature_index(pt, color, sq, king_sq, view)
            for pt, color, sq in pieces]


def phase_scale_from_pieces(pieces: Sequence[tuple[int, int, int]]) -> float:
    minors = sum(1 for pt, _color, _sq in pieces
                 if pt in (KNIGHT, BISHOP))
    rooks = sum(1 for pt, _color, _sq in pieces if pt == ROOK)
    queens = sum(1 for pt, _color, _sq in pieces if pt == QUEEN)
    phase = 3 * minors + 5 * rooks + 10 * queens
    return (128.0 + float(phase)) / 128.0


def load_net(path: str | Path) -> Net:
    data = Path(path).read_bytes()
    if len(data) != NETWORK_SIZE:
        raise ValueError(
            f"{path}: size {len(data)} != expected {NETWORK_SIZE}")

    off = 0

    def take(dtype, count: int):
        nonlocal off
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=off)
        off += arr.nbytes
        return arr.copy()

    iw = take(np.int16, N_FEATURES * N_HIDDEN).reshape(N_FEATURES, N_HIDDEN)
    ib = take(np.int16, N_HIDDEN)
    l1w = take(np.int8, N_L1 * N_L2).reshape(N_L2, N_L1)
    l1b = take(np.int32, N_L2)
    l2w = take(np.float32, N_L2 * N_L3).reshape(N_L3, N_L2)
    l2b = take(np.float32, N_L3)
    ow = take(np.float32, N_L3)
    ob = struct.unpack_from("<f", data, off)[0]
    off += 4
    assert off == len(data)
    return Net(iw, ib, l1w, l1b, l2w, l2b, ow, float(ob))


def write_net(net: Net, path: str | Path) -> None:
    out = Path(path)
    with out.open("wb") as f:
        f.write(np.asarray(net.input_weights, dtype=np.int16).tobytes(order="C"))
        f.write(np.asarray(net.input_biases, dtype=np.int16).tobytes(order="C"))
        f.write(np.asarray(net.l1_weights, dtype=np.int8).tobytes(order="C"))
        f.write(np.asarray(net.l1_biases, dtype=np.int32).tobytes(order="C"))
        f.write(np.asarray(net.l2_weights, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(net.l2_biases, dtype=np.float32).tobytes(order="C"))
        f.write(np.asarray(net.output_weights, dtype=np.float32).tobytes(order="C"))
        f.write(struct.pack("<f", float(net.output_bias)))
    size = out.stat().st_size
    if size != NETWORK_SIZE:
        raise RuntimeError(f"wrote {size} bytes, expected {NETWORK_SIZE}")
