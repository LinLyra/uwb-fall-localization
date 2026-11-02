#!/usr/bin/env python3
"""Paired stream bootstrap and exact randomization test for two submissions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append(
                {
                    **row,
                    "t_start": float(row["t_start"]),
                    "t_end": float(row["t_end"]),
                    "score": float(row["score"]),
                    "t_alarm": float(row["t_alarm"]) if row.get("t_alarm") else None,
                }
            )
        return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--label-a", required=True)
    parser.add_argument("--label-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    pred_a = read_predictions(args.a)
    pred_b = read_predictions(args.b)
    streams = sorted(durations)
    by_gt = {sid: [item for item in ground if item["stream_id"] == sid] for sid in streams}
    by_a = {sid: [item for item in pred_a if item["stream_id"] == sid] for sid in streams}
    by_b = {sid: [item for item in pred_b if item["stream_id"] == sid] for sid in streams}

    def score(predictions: list[dict], local_gt: list[dict], local_durations: dict[str, float]) -> float:
        return float(evaluate(local_gt, predictions, local_durations)["ranking_score"]["f1_mean"])

    score_a = score(pred_a, ground, durations)
    score_b = score(pred_b, ground, durations)
    observed = score_a - score_b
    rng = np.random.default_rng(args.seed)
    bootstrap: list[float] = []
    for _ in range(args.replicates):
        sampled = rng.choice(streams, size=len(streams), replace=True)
        boot_gt: list[dict] = []
        boot_a: list[dict] = []
        boot_b: list[dict] = []
        boot_durations: dict[str, float] = {}
        for index, original_value in enumerate(sampled):
            original = str(original_value)
            clone = f"bootstrap_{index:02d}"
            boot_durations[clone] = durations[original]
            boot_gt.extend({**item, "stream_id": clone} for item in by_gt[original])
            boot_a.extend({**item, "stream_id": clone} for item in by_a[original])
            boot_b.extend({**item, "stream_id": clone} for item in by_b[original])
        bootstrap.append(score(boot_a, boot_gt, boot_durations) - score(boot_b, boot_gt, boot_durations))

    randomized: list[float] = []
    for swaps in itertools.product((False, True), repeat=len(streams)):
        perm_a: list[dict] = []
        perm_b: list[dict] = []
        for stream, swap in zip(streams, swaps):
            perm_a.extend(by_b[stream] if swap else by_a[stream])
            perm_b.extend(by_a[stream] if swap else by_b[stream])
        randomized.append(score(perm_a, ground, durations) - score(perm_b, ground, durations))
    random_array = np.asarray(randomized)
    p_value = float(np.mean(np.abs(random_array) >= abs(observed) - 1e-12))
    boot_array = np.asarray(bootstrap)
    result = {
        "a": args.label_a,
        "b": args.label_b,
        "score_a": score_a,
        "score_b": score_b,
        "paired_difference_a_minus_b": observed,
        "stream_bootstrap_replicates": args.replicates,
        "paired_difference_ci_95": [
            float(np.quantile(boot_array, 0.025)),
            float(np.quantile(boot_array, 0.975)),
        ],
        "exact_randomization_assignments": int(len(random_array)),
        "two_sided_randomization_p": p_value,
        "seed": args.seed,
        "note": "Five streams limit power; effect size and interval are primary.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
