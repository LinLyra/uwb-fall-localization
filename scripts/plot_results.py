#!/usr/bin/env python3
"""Generate repository-ready, white-background result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    output = Path("reports/figures")
    output.mkdir(parents=True, exist_ok=True)
    tuned_names = ["Official\nenergy", "Logistic\nv2", "HistGradient\nBoosting", "Random\nForest"]
    tuned = np.array(
        [
            [0.3151, 0.3383, 0.0597],
            [0.3699, 0.4036, 0.0538],
            [0.6667, 0.7285, 0.1192],
            [0.6912, 0.7500, 0.2059],
        ]
    )
    nested_names = ["RF all\nfeatures", "RF without peak\n(exploratory choice)"]
    nested = np.array([[0.6575, 0.7034, 0.1931], [0.7080, 0.7862, 0.1931]])
    labels = ["Ranking F1", "F1 @ 0.3", "F1 @ 0.7"]
    colors = ["#0F766E", "#38BDF8", "#94A3B8"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), facecolor="white", gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, names, values, title in (
        (axes[0], tuned_names, tuned, "A. Validation-tuned comparison"),
        (axes[1], nested_names, nested, "B. Decoder-nested RF variants"),
    ):
        x = np.arange(len(names))
        width = 0.23
        for metric in range(values.shape[1]):
            ax.bar(x + (metric - 1) * width, values[:, metric], width, label=labels[metric], color=colors[metric])
        ax.set_xticks(x, names)
        ax.set_ylim(0, 0.9)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("UWB fall localization — protocols shown separately", fontsize=16)
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
