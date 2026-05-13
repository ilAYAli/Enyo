#!/usr/bin/env python3
"""Audit NNUE2 target quality before spending time on training."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


MPE_SCALE = 2.5 / 400.0
ABS_BUCKETS = (
    (0.0, 50.0),
    (50.0, 100.0),
    (100.0, 300.0),
    (300.0, 800.0),
    (800.0, 1600.0),
    (1600.0, float("inf")),
)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else float("nan")


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * q)))
    return ordered[idx]


def bucket_name(lo: float, hi: float) -> str:
    if math.isinf(hi):
        return f">={lo:.0f}"
    return f"{lo:.0f}-{hi:.0f}"


def mpe_target(score: float, wdl: float, wdl_lambda: float) -> float:
    cp_target = 1.0 / (1.0 + math.exp(-score * MPE_SCALE))
    return wdl_lambda * cp_target + (1.0 - wdl_lambda) * wdl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path)
    ap.add_argument("--rows", type=int, default=100000,
                    help="Maximum rows to audit. Use 0 for all rows.")
    ap.add_argument("--lambdas", default="1.0,0.9,0.75,0.5",
                    help="Comma-separated MPE WDL lambdas to summarize.")
    args = ap.parse_args()

    rows = 0
    scores: list[float] = []
    wdls: list[float] = []
    source_scores: list[float] = []
    teacher_times: list[float] = []
    by_wdl: dict[float, list[float]] = defaultdict(list)
    side_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    abs_bucket_counts: Counter[str] = Counter()
    source_sign_disagree = 0
    missing_source = 0

    with args.jsonl.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows += 1

            score = float(row["score"])
            wdl = float(row.get("wdl", 0.5))
            scores.append(score)
            wdls.append(wdl)
            by_wdl[wdl].append(score)
            side_counts[str(row.get("side", ""))] += 1
            termination_counts[str(row.get("termination", ""))] += 1
            teacher_times.append(float(row.get("time_s") or 0.0))

            abs_score = abs(score)
            for lo, hi in ABS_BUCKETS:
                if abs_score >= lo and (math.isinf(hi) or abs_score < hi):
                    abs_bucket_counts[bucket_name(lo, hi)] += 1
                    break

            if "source_score" in row:
                source_score = float(row["source_score"])
                source_scores.append(source_score)
                if (score > 0.0) != (source_score > 0.0) and score != 0.0 \
                        and source_score != 0.0:
                    source_sign_disagree += 1
            else:
                missing_source += 1

            if args.rows > 0 and rows >= args.rows:
                break

    abs_scores = [abs(score) for score in scores]
    print(f"rows={rows}")
    print(f"side={dict(side_counts)}")
    print(f"termination={dict(termination_counts)}")
    print(
        "score "
        f"mean={mean(scores):.1f} "
        f"p50={quantile(scores, 0.50):.0f} "
        f"abs_p50={quantile(abs_scores, 0.50):.0f} "
        f"abs_p90={quantile(abs_scores, 0.90):.0f} "
        f"abs_p99={quantile(abs_scores, 0.99):.0f} "
        f"min={min(scores):.0f} max={max(scores):.0f}")
    print(f"abs_buckets={dict(abs_bucket_counts)}")
    print(f"wdl_counts={dict(Counter(wdls))}")
    print(
        "wdl_score_mean="
        + " ".join(
            f"{wdl:g}:{mean(values):.1f}"
            for wdl, values in sorted(by_wdl.items())))
    print(
        "teacher_time "
        f"mean={mean(teacher_times):.4f}s "
        f"p95={quantile(teacher_times, 0.95):.4f}s "
        f"max={max(teacher_times):.4f}s")

    if source_scores:
        score_mean = mean(scores[:len(source_scores)])
        source_mean = mean(source_scores)
        cov = sum(
            (score - score_mean) * (source - source_mean)
            for score, source in zip(scores, source_scores)
        ) / len(source_scores)
        score_var = sum(
            (score - score_mean) ** 2 for score in scores[:len(source_scores)]
        ) / len(source_scores)
        source_var = sum(
            (source - source_mean) ** 2 for source in source_scores
        ) / len(source_scores)
        corr = cov / math.sqrt(max(1e-12, score_var * source_var))
        mae = mean(
            abs(score - source)
            for score, source in zip(scores, source_scores))
        disagree_pct = 100.0 * source_sign_disagree / len(source_scores)
        print(
            "source_vs_teacher "
            f"rows={len(source_scores)} mae={mae:.1f} corr={corr:.3f} "
            f"sign_disagree={disagree_pct:.2f}%")
    if missing_source:
        print(f"missing_source_score={missing_source}")

    lambdas = [float(item) for item in args.lambdas.split(",") if item]
    for wdl_lambda in lambdas:
        targets = [
            mpe_target(score, wdl, wdl_lambda)
            for score, wdl in zip(scores, wdls)
        ]
        print(
            f"mpe_target lambda={wdl_lambda:.2f} "
            f"mean={mean(targets):.3f} "
            f"p05={quantile(targets, 0.05):.3f} "
            f"p50={quantile(targets, 0.50):.3f} "
            f"p95={quantile(targets, 0.95):.3f}")


if __name__ == "__main__":
    main()
