#!/usr/bin/env python3
"""Plot range-shift and sensor-removal controls."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    report_dir = Path("reports/spatial_audit")
    values = json.loads((report_dir / "spatial_controls_summary.json").read_text(encoding="utf-8"))
    offsets = [-24, -16, -8, 0, 8, 16, 24]
    scores = [
        values[f"range_shift_{offset:+d}"]["ranking_f1_mean"] if offset else 0.6574712643678161
        for offset in offsets
    ]
    radar_names = ["Drop R0", "Drop R1", "Drop R2", "Drop strongest", "Best reorder"]
    radar_scores = [
        values["drop_radar_0"]["ranking_f1_mean"],
        values["drop_radar_1"]["ranking_f1_mean"],
        values["drop_radar_2"]["ranking_f1_mean"],
        values["drop_highest_energy"]["ranking_f1_mean"],
        max(value["ranking_f1_mean"] for key, value in values.items() if key.startswith("radar_order_")),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), facecolor="white")
    axes[0].plot(offsets, scores, marker="o", linewidth=2, color="#0F766E")
    axes[0].axhline(0.6574712643678161, color="#94A3B8", linestyle="--", label="Clean reference")
    axes[0].set_xlabel("Range-bin shift")
    axes[0].set_ylabel("Ranking F1")
    axes[0].set_title("A. Absolute range sensitivity")
    axes[0].legend(frameon=False)

    axes[1].bar(radar_names, radar_scores, color="#38BDF8")
    axes[1].axhline(0.6574712643678161, color="#94A3B8", linestyle="--", label="Clean reference")
    axes[1].set_title("B. Radar removal and reordering")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_ylim(0, 0.75)
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Spatial robustness controls", fontsize=15)
    fig.tight_layout()
    fig.savefig(report_dir / "spatial_controls.png", dpi=220, bbox_inches="tight")
    fig.savefig(report_dir / "spatial_controls.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
