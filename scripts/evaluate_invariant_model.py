#!/usr/bin/env python3
"""Nested evaluation of a radar-order invariant Random Forest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_tabular_baselines import decode, load_cache, write_predictions
from uwb_fall.invariance import canonicalize_radar_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--augmentation-cache", type=Path)
    parser.add_argument("--augmentation-prefix")
    parser.add_argument("--augmentation-fraction", type=float, default=1.0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    train = load_cache(args.cache / "train_w100_h25.npz")
    val = load_cache(args.cache / "val_w100_h25.npz")
    train_x = canonicalize_radar_features(train["features"])
    if args.augmentation_cache is not None:
        augmented = load_cache(args.augmentation_cache)
        eligible = np.arange(len(augmented["labels"]))
        if args.augmentation_prefix:
            eligible = eligible[np.char.startswith(augmented["kinds"].astype(str), args.augmentation_prefix)]
        target = min(len(eligible), int(round(len(train["labels"]) * args.augmentation_fraction)))
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(eligible, size=target, replace=False)
        train_x = np.concatenate([train_x, augmented["features"][selected]], axis=0)
        train["labels"] = np.concatenate([train["labels"], augmented["labels"][selected]], axis=0)
    val_x = canonicalize_radar_features(val["features"])
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train_x, train["labels"])
    scores = model.predict_proba(val_x)[:, 1]
    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    combined: list[dict] = []
    folds: dict[str, dict] = {}
    for held_out in sorted(durations):
        selection_mask = val["stream_ids"] != held_out
        held_mask = ~selection_mask
        selection_durations = {key: value for key, value in durations.items() if key != held_out}
        best: tuple[float, dict] | None = None
        for threshold in np.arange(0.35, 0.91, 0.05):
            for merge_gap in (0.0, 0.5, 1.0):
                for trim in (0.0, 0.25, 0.5, 0.75):
                    rows = decode(
                        scores[selection_mask],
                        val["stream_ids"][selection_mask],
                        val["starts"][selection_mask],
                        val["ends"][selection_mask],
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
        combined.extend(held_rows)
        held_report = evaluate(ground, held_rows, {held_out: durations[held_out]})
        folds[held_out] = {
            "selected_decoder": params,
            "selection_ranking_f1": selection_score,
            "held_out_ranking_f1": held_report["ranking_score"]["f1_mean"],
        }
    report = evaluate(ground, combined, durations)
    summary = {
        "representation": "sorted three-radar groups; scalar entropy unchanged",
        "seed": args.seed,
        "augmentation_cache": str(args.augmentation_cache) if args.augmentation_cache else None,
        "augmentation_prefix": args.augmentation_prefix,
        "augmentation_fraction": args.augmentation_fraction if args.augmentation_cache else None,
        "ranking_f1_mean": report["ranking_score"]["f1_mean"],
        "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
        "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
        "folds": folds,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output / "invariant_predictions.csv", combined)
    (args.output / "invariant_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "invariant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
