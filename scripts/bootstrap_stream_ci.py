#!/usr/bin/env python3
"""Stream-level bootstrap confidence intervals for an event prediction file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    with args.predictions.open(newline="", encoding="utf-8") as handle:
        predictions = [
            {
                **row,
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "score": float(row["score"]),
                "t_alarm": float(row["t_alarm"]) if row.get("t_alarm") else None,
            }
            for row in csv.DictReader(handle)
        ]
    stream_ids = sorted(durations)
    by_gt = {sid: [item for item in ground if item["stream_id"] == sid] for sid in stream_ids}
    by_pred = {sid: [item for item in predictions if item["stream_id"] == sid] for sid in stream_ids}
    rng = np.random.default_rng(args.seed)
    values: list[float] = []
    for _ in range(args.replicates):
        sampled = rng.choice(stream_ids, size=len(stream_ids), replace=True)
        boot_gt: list[dict] = []
        boot_pred: list[dict] = []
        boot_durations: dict[str, float] = {}
        for index, original in enumerate(sampled):
            clone = f"bootstrap_{index:02d}"
            boot_durations[clone] = durations[str(original)]
            boot_gt.extend({**item, "stream_id": clone} for item in by_gt[str(original)])
            boot_pred.extend({**item, "stream_id": clone} for item in by_pred[str(original)])
        values.append(float(evaluate(boot_gt, boot_pred, boot_durations)["ranking_score"]["f1_mean"]))
    array = np.asarray(values)
    result = {
        "unit": "validation stream",
        "replicates": args.replicates,
        "seed": args.seed,
        "estimate": float(evaluate(ground, predictions, durations)["ranking_score"]["f1_mean"]),
        "bootstrap_mean": float(np.mean(array)),
        "ci_95_percentile": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
        "warning": "Decoder hyperparameters were selected on this validation set; this interval measures stream variability, not an unbiased generalization interval.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
