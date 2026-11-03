#!/usr/bin/env python3
"""Evaluate range-position and radar-view shortcuts without model refitting."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_tabular_baselines import decode, load_cache, write_predictions
from uwb_fall.features import power_features, shift_range
from uwb_fall.invariance import canonicalize_radar_features


def transform_features(
    streams_dir: Path,
    stream_ids: np.ndarray,
    starts: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    offsets = (-24, -16, -8, 8, 16, 24)
    mask_fractions = (0.125, 0.25, 0.5)
    permutations = [item for item in itertools.permutations(range(3)) if item != (0, 1, 2)]
    names = (
        [f"range_shift_{offset:+d}" for offset in offsets]
        + [f"range_mask_{fraction:.3f}" for fraction in mask_fractions]
        + [f"drop_radar_{index}" for index in range(3)]
        + ["drop_highest_energy"]
        + ["radar_order_" + "".join(map(str, order)) for order in permutations]
    )
    collected: dict[str, list[np.ndarray]] = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    current = None
    handle = None
    try:
        for stream_id, start in zip(stream_ids, starts):
            if stream_id != current:
                if handle is not None:
                    handle.close()
                index = str(stream_id).rsplit("_", 1)[-1]
                handle = h5py.File(streams_dir / f"stream_{index}.h5", "r")
                current = stream_id
            assert handle is not None
            first = int(round(float(start) * 50.0))
            values = handle["cir"][:, first : first + 100, :].astype(np.float64)
            bins = values.shape[2] // 2
            i = values[:, :, :bins]
            q = values[:, :, bins:]
            i -= np.mean(i, axis=1, keepdims=True)
            q -= np.mean(q, axis=1, keepdims=True)
            power = i * i + q * q

            for offset in offsets:
                collected[f"range_shift_{offset:+d}"].append(power_features(shift_range(power, offset)))
            for fraction in mask_fractions:
                width = int(round(bins * fraction))
                mask_start = int(rng.integers(0, bins - width + 1))
                masked = power.copy()
                masked[:, :, mask_start : mask_start + width] = 0.0
                collected[f"range_mask_{fraction:.3f}"].append(power_features(masked))
            for radar in range(3):
                dropped = power.copy()
                dropped[radar] = 0.0
                collected[f"drop_radar_{radar}"].append(power_features(dropped))
            strongest = int(np.argmax(np.mean(power, axis=(1, 2))))
            dropped = power.copy()
            dropped[strongest] = 0.0
            collected["drop_highest_energy"].append(power_features(dropped))
            for order in permutations:
                collected["radar_order_" + "".join(map(str, order))].append(power_features(power[list(order)]))
    finally:
        if handle is not None:
            handle.close()
    return {name: np.stack(rows) for name, rows in collected.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--nested-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--augmentation-cache", type=Path)
    parser.add_argument("--augmentation-prefix")
    parser.add_argument("--augmentation-fraction", type=float, default=0.25)
    parser.add_argument("--canonicalize-radars", action="store_true")
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
    train_x = canonicalize_radar_features(train["features"]) if args.canonicalize_radars else train["features"]
    train_y = train["labels"]
    if args.augmentation_cache:
        augmented = load_cache(args.augmentation_cache)
        eligible = np.arange(len(augmented["labels"]))
        if args.augmentation_prefix:
            eligible = eligible[np.char.startswith(augmented["kinds"].astype(str), args.augmentation_prefix)]
        count = min(len(eligible), int(round(len(train_y) * args.augmentation_fraction)))
        selected = np.random.default_rng(args.seed).choice(eligible, size=count, replace=False)
        train_x = np.concatenate([train_x, augmented["features"][selected]])
        train_y = np.concatenate([train_y, augmented["labels"][selected]])
    model.fit(train_x, train_y)
    transformed = transform_features(
        args.package / "streams" / "val", val["stream_ids"], val["starts"], args.seed
    )
    nested_data = json.loads(args.nested_summary.read_text(encoding="utf-8"))
    folds = nested_data.get("all", nested_data)["folds"]
    durations = stream_durations(args.package / "streams" / "val")
    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for name, features in transformed.items():
        if args.canonicalize_radars:
            features = canonicalize_radar_features(features)
        scores = model.predict_proba(features)[:, 1]
        rows: list[dict] = []
        for stream_id in sorted(set(val["stream_ids"].tolist())):
            mask = val["stream_ids"] == stream_id
            params = folds[stream_id]["selected_decoder"]
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
        report = evaluate(ground, rows, durations)
        summary[name] = {
            "ranking_f1_mean": report["ranking_score"]["f1_mean"],
            "f1_at_0.3": report["by_theta"]["0.3"]["f1"],
            "f1_at_0.7": report["by_theta"]["0.7"]["f1"],
            "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
            "prediction_count": len(rows),
        }
        write_predictions(args.output / f"{name}_predictions.csv", rows)
        print(name, json.dumps(summary[name], sort_keys=True))
    (args.output / "spatial_controls_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
