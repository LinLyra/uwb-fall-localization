#!/usr/bin/env python3
"""Train reproducible tabular baselines and tune only event decoding on validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def decode(
    scores: np.ndarray,
    stream_ids: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    threshold: float,
    merge_gap: float,
    trim: float,
) -> list[dict]:
    predictions: list[dict] = []
    for stream_id in sorted(set(stream_ids.tolist())):
        indices = np.flatnonzero((stream_ids == stream_id) & (scores >= threshold))
        if len(indices) == 0:
            continue
        groups: list[list[int]] = [[int(indices[0])]]
        for index in indices[1:]:
            if float(starts[index]) - float(ends[groups[-1][-1]]) <= merge_gap:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        for group in groups:
            start = max(0.0, float(starts[group[0]]) + trim)
            end = float(ends[group[-1]]) - trim
            if end <= start:
                continue
            predictions.append(
                {
                    "stream_id": stream_id,
                    "t_start": start,
                    "t_end": end,
                    "score": float(np.max(scores[group])),
                    "t_alarm": end,
                }
            )
    return predictions


def write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stream_id", "t_start", "t_end", "score", "t_alarm"])
        writer.writeheader()
        writer.writerows(rows)


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
    x_train, y_train = train["features"], train["labels"]
    positive_weight = float((len(y_train) - y_train.sum()) / max(int(y_train.sum()), 1))
    sample_weight = np.where(y_train == 1, positive_weight, 1.0)

    models = {
        "logistic_v2": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=2027),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07,
            max_iter=180,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=2027,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=2027,
        ),
    }
    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    summary: dict[str, dict] = {}
    args.output.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if name == "hist_gradient_boosting":
            model.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(x_train, y_train)
        scores = model.predict_proba(val["features"])[:, 1]
        best: tuple[float, dict, list[dict], dict] | None = None
        trials: list[dict] = []
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
                    params = {
                        "threshold": round(float(threshold), 2),
                        "merge_gap": merge_gap,
                        "trim": trim,
                    }
                    trials.append({**params, "ranking_f1_mean": rank})
                    if best is None or rank > best[0]:
                        best = (rank, params, rows, report)
        assert best is not None
        _, params, rows, report = best
        write_predictions(args.output / f"{name}_val_predictions.csv", rows)
        (args.output / f"{name}_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        summary[name] = {
            "decoder": params,
            "ranking_f1_mean": report["ranking_score"]["f1_mean"],
            "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
            "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
            "prediction_count": len(rows),
            "validation_trials": len(trials),
        }
        print(name, json.dumps(summary[name], sort_keys=True))

    (args.output / "tabular_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
