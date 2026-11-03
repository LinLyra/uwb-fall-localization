#!/usr/bin/env python3
"""Analytic and simulated sensitivity of tIoU to boundary error."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def temporal_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    overlap = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    return overlap / union if union > 0 else 0.0


def allowed_error(duration: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    """Maximum non-negative error for three symmetric boundary patterns."""
    if mode == "outward":
        return duration * (1.0 / threshold - 1.0) / 2.0
    if mode == "inward":
        return duration * (1.0 - threshold) / 2.0
    if mode == "translation":
        return duration * (1.0 - threshold) / (1.0 + threshold)
    raise ValueError(f"unknown mode: {mode}")


def read_evaluable_durations(path: Path, split: str) -> np.ndarray:
    durations = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["stream_id"].startswith(f"{split}_"):
                continue
            flagged = row.get("quality") == "low" or row.get("needs_review", "").lower() == "true"
            ignored = row.get("ignore", "").lower() == "true"
            if ignored or flagged:
                continue
            durations.append(float(row["t_end"]) - float(row["t_start"]))
    return np.asarray(durations, dtype=np.float64)


def simulate_noise(
    durations: np.ndarray,
    half_width: float,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    rates = {threshold: [] for threshold in (0.3, 0.5, 0.7)}
    for _ in range(replicates):
        start_error = rng.uniform(-half_width, half_width, size=len(durations))
        end_error = rng.uniform(-half_width, half_width, size=len(durations))
        ious = []
        for duration, left, right in zip(durations, start_error, end_error):
            predicted_end = max(1e-6, float(duration + right))
            ious.append(temporal_iou(0.0, float(duration), float(left), predicted_end))
        ious_array = np.asarray(ious)
        for threshold in rates:
            rates[threshold].append(float(np.mean(ious_array >= threshold)))
    return {
        f"fraction_at_tiou_{threshold:.1f}_mean": float(np.mean(values))
        for threshold, values in rates.items()
    } | {
        f"fraction_at_tiou_{threshold:.1f}_ci95": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for threshold, values in rates.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    durations = read_evaluable_durations(args.events, "val")
    rng = np.random.default_rng(args.seed)
    quantile_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    quantiles = np.quantile(durations, quantile_levels)
    result = {
        "event_count": int(len(durations)),
        "duration_seconds": {
            f"q{int(level * 100):02d}": float(value)
            for level, value in zip(quantile_levels, quantiles)
        },
        "median_event_allowed_error_seconds": {
            f"tiou_{threshold:.1f}": {
                mode: float(allowed_error(np.asarray([np.median(durations)]), threshold, mode)[0])
                for mode in ("outward", "inward", "translation")
            }
            for threshold in (0.3, 0.5, 0.7)
        },
        "uniform_independent_boundary_noise": {
            "plus_minus_0.5_seconds": simulate_noise(durations, 0.5, args.replicates, rng),
            "plus_minus_1.0_seconds": simulate_noise(durations, 1.0, args.replicates, rng),
        },
        "simulation": {
            "replicates": args.replicates,
            "seed": args.seed,
            "warning": "Sensitivity analysis under independent uniform perturbations, not an empirical annotation-agreement estimate.",
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "boundary_limits.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    duration_grid = np.linspace(max(0.25, float(durations.min())), float(durations.max()), 200)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), facecolor="white")
    for threshold, color in zip((0.3, 0.5, 0.7), ("#38BDF8", "#0F766E", "#F59E0B")):
        axes[0].plot(
            duration_grid,
            allowed_error(duration_grid, threshold, "translation"),
            color=color,
            label=f"tIoU ≥ {threshold:.1f}",
        )
    axes[0].axvline(float(np.median(durations)), color="#94A3B8", linestyle="--", label="Median duration")
    axes[0].set_xlabel("Event duration (s)")
    axes[0].set_ylabel("Maximum translation error (s)")
    axes[0].set_title("A. Analytic tolerance")
    axes[0].legend(frameon=False)

    noise_labels = ["±0.5 s", "±1.0 s"]
    x = np.arange(2)
    width = 0.23
    simulations = result["uniform_independent_boundary_noise"]
    for index, (threshold, color) in enumerate(zip((0.3, 0.5, 0.7), ("#38BDF8", "#0F766E", "#F59E0B"))):
        values = [
            simulations["plus_minus_0.5_seconds"][f"fraction_at_tiou_{threshold:.1f}_mean"],
            simulations["plus_minus_1.0_seconds"][f"fraction_at_tiou_{threshold:.1f}_mean"],
        ]
        axes[1].bar(x + (index - 1) * width, values, width, color=color, label=f"tIoU ≥ {threshold:.1f}")
    axes[1].set_xticks(x, noise_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Expected fraction above threshold")
    axes[1].set_title("B. Boundary-noise sensitivity")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"tIoU boundary sensitivity ({len(durations)} evaluable events)", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output / "boundary_limits.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "boundary_limits.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
