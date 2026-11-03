#!/usr/bin/env python3
"""Compare fall-to-join proximity with matched random and ADL baselines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from train_tabular_baselines import load_cache


def read_joins(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"].startswith("val_"):
                rows.setdefault(row["stream_id"], []).append(float(row["t_start"]))
    return {stream: np.asarray(sorted(values)[1:]) for stream, values in rows.items()}


def read_events(path: Path) -> dict[str, np.ndarray]:
    rows: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["stream_id"].startswith("val_"):
                continue
            flagged = row.get("quality") == "low" or row.get("needs_review", "").lower() == "true"
            ignored = row.get("ignore", "").lower() == "true"
            if flagged or ignored:
                continue
            rows.setdefault(row["stream_id"], []).append((float(row["t_start"]), float(row["t_end"])))
    return {stream: np.asarray(values) for stream, values in rows.items()}


def nearest(points: np.ndarray, joins: np.ndarray) -> np.ndarray:
    return np.min(np.abs(points[:, None] - joins[None, :]), axis=1)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "median_s": float(np.median(values)),
        "fraction_within_2s": float(np.mean(values <= 2.0)),
        "fraction_within_5s": float(np.mean(values <= 5.0)),
        "fraction_within_10s": float(np.mean(values <= 10.0)),
    }


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
    joins = read_joins(args.composition)
    events = read_events(args.events)
    cache = load_cache(args.cache / "val_w100_h25.npz")
    centres = (cache["starts"] + cache["ends"]) / 2.0

    fall_distances = []
    adl_pools: dict[str, np.ndarray] = {}
    durations: dict[str, float] = {}
    for stream, intervals in events.items():
        event_centres = np.mean(intervals, axis=1)
        fall_distances.extend(nearest(event_centres, joins[stream]))
        stream_mask = cache["stream_ids"] == stream
        stream_centres = centres[stream_mask]
        safe = np.ones(len(stream_centres), dtype=bool)
        for start, end in intervals:
            safe &= ~((stream_centres >= start - 5.0) & (stream_centres <= end + 5.0))
        adl_pools[stream] = stream_centres[safe]
        durations[stream] = float(cache["ends"][stream_mask].max())
    fall = np.asarray(fall_distances)

    random_stats = []
    adl_stats = []
    pooled_random = []
    pooled_adl = []
    for _ in range(args.replicates):
        random_distances = []
        adl_distances = []
        for stream, intervals in events.items():
            count = len(intervals)
            random_points = rng.uniform(0.0, durations[stream], size=count)
            adl_points = rng.choice(adl_pools[stream], size=count, replace=False)
            random_distances.extend(nearest(random_points, joins[stream]))
            adl_distances.extend(nearest(adl_points, joins[stream]))
        random_array = np.asarray(random_distances)
        adl_array = np.asarray(adl_distances)
        random_stats.append(summarize(random_array))
        adl_stats.append(summarize(adl_array))
        if len(pooled_random) < 20_000:
            pooled_random.extend(random_array)
            pooled_adl.extend(adl_array)

    def baseline_report(stats: list[dict[str, float]]) -> dict:
        result = {}
        for key in stats[0]:
            values = np.asarray([item[key] for item in stats])
            result[key] = {
                "mean": float(np.mean(values)),
                "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            }
        return result

    observed = summarize(fall)
    random_medians = np.asarray([item["median_s"] for item in random_stats])
    adl_medians = np.asarray([item["median_s"] for item in adl_stats])
    result = {
        "fall_event_count": int(len(fall)),
        "fall": observed,
        "matched_random_time": baseline_report(random_stats),
        "matched_adl_window": baseline_report(adl_stats),
        "one_sided_median_proximity_p": {
            "fall_vs_random": float((1 + np.sum(random_medians <= observed["median_s"])) / (args.replicates + 1)),
            "fall_vs_adl": float((1 + np.sum(adl_medians <= observed["median_s"])) / (args.replicates + 1)),
        },
        "replicates": args.replicates,
        "seed": args.seed,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "join_proximity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.5, 4.8), facecolor="white")
    for values, label, color in (
        (fall, "Fall centres", "#f59e0b"),
        (np.asarray(pooled_random), "Matched random time", "#38bdf8"),
        (np.asarray(pooled_adl), "Matched ADL windows", "#0f766e"),
    ):
        ordered = np.sort(values)
        ax.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered), where="post", label=label, color=color, linewidth=2)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Distance to nearest synthetic join (s)")
    ax.set_ylabel("Empirical cumulative fraction")
    ax.set_title("Are fall events unusually close to stream joins?")
    ax.grid(color="#e2e8f0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output / "join_proximity.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "join_proximity.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
