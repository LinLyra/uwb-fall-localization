#!/usr/bin/env python3
"""Feature-group ablation using one fixed model family and official metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_tabular_baselines import decode, load_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    train = load_cache(args.cache / "train_w100_h25.npz")
    val = load_cache(args.cache / "val_w100_h25.npz")
    # Layout from window_features_v2: temporal 12, velocity 12, range 48,
    # radar fractions 3, entropy 1, peak motion 6.
    groups = {
        "all": np.arange(82),
        "without_range": np.r_[0:24, 72:82],
        "without_cross_radar": np.r_[0:72, 76:82],
        "without_peak_motion": np.arange(76),
        "temporal_only": np.arange(24),
        "range_only": np.arange(24, 72),
    }
    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    summary: dict[str, dict] = {}
    for name, columns in groups.items():
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=2027,
        )
        model.fit(train["features"][:, columns], train["labels"])
        scores = model.predict_proba(val["features"][:, columns])[:, 1]
        best: tuple[float, dict, dict] | None = None
        for threshold in np.arange(0.35, 0.91, 0.05):
            for merge_gap in (0.0, 0.5, 1.0):
                for trim in (0.0, 0.25, 0.5, 0.75):
                    rows = decode(
                        scores,
                        val["stream_ids"],
                        val["starts"],
                        val["ends"],
                        float(threshold),
                        merge_gap,
                        trim,
                    )
                    report = evaluate(ground, rows, durations)
                    rank = float(report["ranking_score"]["f1_mean"])
                    if best is None or rank > best[0]:
                        best = (
                            rank,
                            {"threshold": round(float(threshold), 2), "merge_gap": merge_gap, "trim": trim},
                            report,
                        )
        assert best is not None
        rank, decoder, report = best
        summary[name] = {
            "feature_count": int(len(columns)),
            "decoder": decoder,
            "ranking_f1_mean": rank,
            "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
            "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
        }
        print(name, json.dumps(summary[name], sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
