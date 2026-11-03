#!/usr/bin/env python3
"""Plot fair missing-radar baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--dropout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    clean = json.loads(args.clean.read_text(encoding="utf-8"))
    dropout = json.loads(args.dropout.read_text(encoding="utf-8"))
    labels = ["Radar 0", "Radar 1", "Radar 2"]
    zero = [clean[f"drop_radar_{i}"]["ranking_f1_mean"] for i in range(3)]
    mean = [clean[f"mean_impute_radar_{i}"]["ranking_f1_mean"] for i in range(3)]
    robust = [dropout[f"drop_radar_{i}"]["ranking_f1_mean"] for i in range(3)]
    x = np.arange(3)
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width, zero, width, label="Zero fill", color="#94a3b8")
    ax.bar(x, mean, width, label="Mean imputation", color="#38bdf8")
    ax.bar(x + width, robust, width, label="Dropout-trained", color="#0f766e")
    ax.axhline(0.6619, color="#f59e0b", linestyle="--", linewidth=1.5, label="Clean model")
    ax.set(xticks=x, xticklabels=labels, ylabel="Ranking F1", ylim=(0, 0.72))
    ax.set_title("Single-radar failure: fair inference and training baselines")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
