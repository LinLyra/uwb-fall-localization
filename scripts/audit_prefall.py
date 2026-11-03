#!/usr/bin/env python3
"""Test whether pre-fall windows predict a future annotated fall."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from uwb_fall.invariance import canonicalize_radar_features


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def read_events(path: Path, split: str, ignore_flagged: bool) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["stream_id"].startswith(f"{split}_"):
                continue
            flagged = row.get("quality") == "low" or row.get("needs_review", "").lower() == "true"
            ignored = row.get("ignore", "").lower() == "true" or (ignore_flagged and flagged)
            if ignored:
                continue
            result.setdefault(row["stream_id"], []).append(
                {"start": float(row["t_start"]), "end": float(row["t_end"])}
            )
    return result


def make_examples(
    cache: dict[str, np.ndarray],
    events: dict[str, list[dict]],
    rng: np.random.Generator,
    lead_s: float = 0.5,
    exclusion_s: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    for stream_id, stream_events in sorted(events.items()):
        indices = np.flatnonzero(cache["stream_ids"] == stream_id)
        if len(indices) == 0:
            continue
        ends = cache["ends"][indices]
        positive_indices: list[int] = []
        for event in stream_events:
            target = event["start"] - lead_s
            local = int(np.argmin(np.abs(ends - target)))
            if abs(float(ends[local]) - target) <= 0.26:
                positive_indices.append(int(indices[local]))

        negative_pool: list[int] = []
        for index in indices:
            start = float(cache["starts"][index])
            end = float(cache["ends"][index])
            too_close = any(
                end >= event["start"] - exclusion_s and start <= event["end"] + exclusion_s
                for event in stream_events
            )
            if not too_close:
                negative_pool.append(int(index))
        if len(negative_pool) < len(positive_indices):
            raise ValueError(f"not enough matched negatives for {stream_id}")
        negatives = rng.choice(negative_pool, size=len(positive_indices), replace=False)
        for index in positive_indices:
            features.append(cache["features"][index])
            labels.append(1)
            groups.append(stream_id)
        for index in negatives:
            features.append(cache["features"][index])
            labels.append(0)
            groups.append(stream_id)
    return np.stack(features), np.asarray(labels, dtype=np.uint8), np.asarray(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--canonicalize-radars", action="store_true")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    train_cache = load_cache(args.cache / "train_w100_h25.npz")
    val_cache = load_cache(args.cache / "val_w100_h25.npz")
    train_events = read_events(args.events, "train", ignore_flagged=False)
    val_events = read_events(args.events, "val", ignore_flagged=True)
    train_x, train_y, _ = make_examples(train_cache, train_events, rng)
    val_x, val_y, val_groups = make_examples(val_cache, val_events, rng)
    if args.canonicalize_radars:
        train_x = canonicalize_radar_features(train_x)
        val_x = canonicalize_radar_features(val_x)

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(train_x, train_y)
    scores = model.predict_proba(val_x)[:, 1]
    estimate = float(roc_auc_score(val_y, scores))
    streams = sorted(set(val_groups.tolist()))
    per_stream = {
        stream: float(roc_auc_score(val_y[val_groups == stream], scores[val_groups == stream]))
        for stream in streams
    }

    bootstrap: list[float] = []
    for _ in range(args.replicates):
        sampled = rng.choice(streams, size=len(streams), replace=True)
        chosen = np.concatenate([np.flatnonzero(val_groups == stream) for stream in sampled])
        bootstrap.append(float(roc_auc_score(val_y[chosen], scores[chosen])))
    values = np.asarray(bootstrap)
    result = {
        "protocol": {
            "window_s": 2.0,
            "lead_s": 0.5,
            "negative_exclusion_s": 5.0,
            "train_positive": int(train_y.sum()),
            "train_negative": int((train_y == 0).sum()),
            "validation_positive": int(val_y.sum()),
            "validation_negative": int((val_y == 0).sum()),
            "bootstrap_unit": "stream",
            "bootstrap_replicates": args.replicates,
            "seed": args.seed,
            "radar_canonicalization": args.canonicalize_radars,
        },
        "roc_auc": estimate,
        "stream_bootstrap_ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
        "per_stream_roc_auc": per_stream,
        "decision": "reject_H2" if estimate >= 0.70 and np.quantile(values, 0.025) > 0.50 else "do_not_reject_H2",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
