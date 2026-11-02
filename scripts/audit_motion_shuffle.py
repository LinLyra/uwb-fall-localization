#!/usr/bin/env python3
"""Measure localization after destroying within-window temporal coherence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_tabular_baselines import decode, load_cache, write_predictions
from uwb_fall.features import power_features


def shuffled_features(
    streams_dir: Path,
    stream_ids: np.ndarray,
    starts: np.ndarray,
    *,
    window: int,
    rate_hz: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # One fixed, independently permuted time map per radar and range bin. Using
    # the same map for every window avoids injecting event-dependent randomness.
    permutation = np.argsort(rng.random((3, 184, window)), axis=2)
    output: list[np.ndarray] = []
    open_stream = None
    handle = None
    try:
        for stream_id, start in zip(stream_ids, starts):
            if stream_id != open_stream:
                if handle is not None:
                    handle.close()
                index = str(stream_id).rsplit("_", 1)[-1]
                handle = h5py.File(streams_dir / f"stream_{index}.h5", "r")
                open_stream = stream_id
            assert handle is not None
            first = int(round(float(start) * rate_hz))
            values = handle["cir"][:, first : first + window, :].astype(np.float64)
            bins = values.shape[2] // 2
            i = values[:, :, :bins]
            q = values[:, :, bins:]
            i -= np.mean(i, axis=1, keepdims=True)
            q -= np.mean(q, axis=1, keepdims=True)
            power = i * i + q * q
            transposed = np.transpose(power, (0, 2, 1))
            shuffled = np.take_along_axis(transposed, permutation, axis=2)
            output.append(power_features(np.transpose(shuffled, (0, 2, 1))))
    finally:
        if handle is not None:
            handle.close()
    return np.stack(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--nested-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    train = load_cache(args.cache / "train_w100_h25.npz")
    val = load_cache(args.cache / "val_w100_h25.npz")
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train["features"], train["labels"])
    transformed = shuffled_features(
        args.package / "streams" / "val",
        val["stream_ids"],
        val["starts"],
        window=100,
        rate_hz=50.0,
        seed=args.seed,
    )
    scores = model.predict_proba(transformed)[:, 1]
    nested = json.loads(args.nested_summary.read_text(encoding="utf-8"))["all"]["folds"]
    rows: list[dict] = []
    for stream_id in sorted(set(val["stream_ids"].tolist())):
        mask = val["stream_ids"] == stream_id
        params = nested[stream_id]["selected_decoder"]
        rows.extend(
            decode(
                scores[mask],
                val["stream_ids"][mask],
                val["starts"][mask],
                val["ends"][mask],
                params["threshold"],
                params["merge_gap"],
                params["trim"],
            )
        )
    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    report = evaluate(ground, rows, durations)
    result = {
        "transform": "fixed independent time permutation per radar and range bin within each 2-second window",
        "seed": args.seed,
        "feature_count": int(transformed.shape[1]),
        "prediction_count": len(rows),
        "ranking_f1_mean": report["ranking_score"]["f1_mean"],
        "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
        "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output / "motion_shuffled_predictions.csv", rows)
    (args.output / "motion_shuffled_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "motion_shuffled_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
