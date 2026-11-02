#!/usr/bin/env python3
"""Leave-one-validation-stream-out selection of event-decoder parameters."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_tabular_baselines import decode, load_cache, write_predictions


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
    all_durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    variants = {"all": np.arange(82), "without_peak_motion": np.arange(76)}
    final: dict[str, dict] = {}

    for variant, columns in variants.items():
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
        combined: list[dict] = []
        folds: dict[str, dict] = {}
        for held_out in sorted(all_durations):
            select_mask = val["stream_ids"] != held_out
            held_mask = ~select_mask
            selection_durations = {key: value for key, value in all_durations.items() if key != held_out}
            best: tuple[float, dict] | None = None
            for threshold in np.arange(0.35, 0.91, 0.05):
                for merge_gap in (0.0, 0.5, 1.0):
                    for trim in (0.0, 0.25, 0.5, 0.75):
                        rows = decode(
                            scores[select_mask],
                            val["stream_ids"][select_mask],
                            val["starts"][select_mask],
                            val["ends"][select_mask],
                            float(threshold),
                            merge_gap,
                            trim,
                        )
                        rank = float(evaluate(ground, rows, selection_durations)["ranking_score"]["f1_mean"])
                        params = {
                            "threshold": round(float(threshold), 2),
                            "merge_gap": merge_gap,
                            "trim": trim,
                        }
                        if best is None or rank > best[0]:
                            best = (rank, params)
            assert best is not None
            selection_score, params = best
            held_rows = decode(
                scores[held_mask],
                val["stream_ids"][held_mask],
                val["starts"][held_mask],
                val["ends"][held_mask],
                params["threshold"],
                params["merge_gap"],
                params["trim"],
            )
            held_report = evaluate(ground, held_rows, {held_out: all_durations[held_out]})
            combined.extend(held_rows)
            folds[held_out] = {
                "selected_decoder": params,
                "selection_ranking_f1": selection_score,
                "held_out_ranking_f1": held_report["ranking_score"]["f1_mean"],
                "held_out_prediction_count": len(held_rows),
            }
        report = evaluate(ground, combined, all_durations)
        write_predictions(args.output / f"nested_{variant}_predictions.csv", combined)
        (args.output / f"nested_{variant}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        final[variant] = {
            "ranking_f1_mean": report["ranking_score"]["f1_mean"],
            "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
            "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
            "folds": folds,
        }
        print(variant, json.dumps(final[variant], sort_keys=True))
    (args.output / "nested_decoder_summary.json").write_text(json.dumps(final, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
