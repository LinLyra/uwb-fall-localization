#!/usr/bin/env python3
"""Test whether synthetic stream joins are detectable from CIR features."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from train_tabular_baselines import load_cache
from uwb_fall.invariance import canonicalize_radar_features


def read_boundaries(path: Path, split: str) -> dict[str, list[float]]:
    by_stream: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"].startswith(f"{split}_"):
                by_stream.setdefault(row["stream_id"], []).append(row)
    result = {}
    for stream, rows in by_stream.items():
        rows.sort(key=lambda row: float(row["t_start"]))
        result[stream] = [float(row["t_start"]) for row in rows[1:]]
    return result


def make_examples(
    cache: dict[str, np.ndarray],
    boundaries: dict[str, list[float]],
    rng: np.random.Generator,
    exclusion_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[int] = []
    groups: list[str] = []
    centers = (cache["starts"] + cache["ends"]) / 2.0
    for stream, joins in sorted(boundaries.items()):
        indices = np.flatnonzero(cache["stream_ids"] == stream)
        positive: list[int] = []
        for join in joins:
            local = indices[int(np.argmin(np.abs(centers[indices] - join)))]
            if abs(float(centers[local]) - join) <= 0.26:
                positive.append(int(local))
        negative_pool = [
            int(index)
            for index in indices
            if all(abs(float(centers[index]) - join) >= exclusion_s for join in joins)
        ]
        count = min(len(positive), len(negative_pool))
        negative = rng.choice(negative_pool, size=count, replace=False)
        for index in positive[:count]:
            x.append(cache["features"][index]); y.append(1); groups.append(stream)
        for index in negative:
            x.append(cache["features"][index]); y.append(0); groups.append(stream)
    return canonicalize_radar_features(np.stack(x)), np.asarray(y), np.asarray(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--composition", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    train_boundaries = read_boundaries(args.composition, "train")
    val_boundaries = read_boundaries(args.composition, "val")
    train = load_cache(args.cache / "train_w100_h25.npz")
    val = load_cache(args.cache / "val_w100_h25.npz")
    train_x, train_y, _ = make_examples(train, train_boundaries, rng)
    val_x, val_y, groups = make_examples(val, val_boundaries, rng)
    model = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=args.seed,
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(val_x)[:, 1]
    estimate = float(roc_auc_score(val_y, scores))
    streams = sorted(set(groups.tolist()))
    per_stream = {
        stream: float(roc_auc_score(val_y[groups == stream], scores[groups == stream]))
        for stream in streams
    }
    bootstrap = []
    for _ in range(args.replicates):
        sampled = rng.choice(streams, size=len(streams), replace=True)
        chosen = np.concatenate([np.flatnonzero(groups == stream) for stream in sampled])
        bootstrap.append(float(roc_auc_score(val_y[chosen], scores[chosen])))
    values = np.asarray(bootstrap)
    event_distances = []
    with args.events.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stream = row["stream_id"]
            if not stream.startswith("val_"):
                continue
            flagged = row.get("quality") == "low" or row.get("needs_review", "").lower() == "true"
            if flagged or row.get("ignore", "").lower() == "true":
                continue
            centre = (float(row["t_start"]) + float(row["t_end"])) / 2.0
            event_distances.append(min(abs(centre - join) for join in val_boundaries[stream]))
    distances = np.asarray(event_distances)
    result = {
        "roc_auc": estimate,
        "stream_bootstrap_ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "train_positive": int(train_y.sum()),
        "train_negative": int((train_y == 0).sum()),
        "validation_positive": int(val_y.sum()),
        "validation_negative": int((val_y == 0).sum()),
        "per_stream_roc_auc": per_stream,
        "window_s": 2.0,
        "negative_exclusion_s": 5.0,
        "seed": args.seed,
        "evaluable_event_distance_to_nearest_join_s": {
            "median": float(np.median(distances)),
            "q90": float(np.quantile(distances, 0.9)),
            "fraction_within_2s": float(np.mean(distances <= 2.0)),
            "fraction_within_5s": float(np.mean(distances <= 5.0)),
            "fraction_within_10s": float(np.mean(distances <= 10.0)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
