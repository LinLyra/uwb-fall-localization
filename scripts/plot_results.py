#!/usr/bin/env python3
"""Generate repository-ready, white-background result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output = Path("reports/figures")
    output.mkdir(parents=True, exist_ok=True)
    names = ["Official\nenergy", "Logistic\nv2", "HistGradient\nBoosting", "Random\nForest", "RF nested\nno peak"]
    scores = np.array([0.3151, 0.3699, 0.6667, 0.6912, 0.7080])
    lower = np.array([np.nan, 0.2708, 0.5630, 0.5783, np.nan])
    upper = np.array([np.nan, 0.4456, 0.7508, 0.7398, np.nan])
    colors = ["#94A3B8", "#60A5FA", "#2DD4BF", "#14B8A6", "#0F766E"]

    fig, ax = plt.subplots(figsize=(9.2, 5.2), facecolor="white")
    x = np.arange(len(names))
    bars = ax.bar(x, scores, color=colors, width=0.68)
    valid = np.isfinite(lower)
    ax.errorbar(
        x[valid],
        scores[valid],
        yerr=np.vstack([scores[valid] - lower[valid], upper[valid] - scores[valid]]),
        fmt="none",
        ecolor="#334155",
        capsize=5,
        linewidth=1.5,
    )
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.018, f"{score:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("Official ranking F1 (mean at tIoU 0.1/0.3/0.5)")
    ax.set_title("UWB fall localization: subject-disjoint validation")
    ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
