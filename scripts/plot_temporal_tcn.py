#!/usr/bin/env python3
"""Plot temporal-model shortcut controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tcn", type=Path, required=True)
    parser.add_argument("--rf-clean", type=float, default=0.6575)
    parser.add_argument("--rf-shuffled", type=float, default=0.6308)
    parser.add_argument("--rf-prefall", type=float, default=0.9407)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tcn = json.loads(args.tcn.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    x = np.arange(2)
    width = 0.34
    axes[0].bar(x - width / 2, [args.rf_clean, args.rf_shuffled], width, label="Random Forest", color="#64748b")
    axes[0].bar(x + width / 2, [tcn["clean"]["ranking_f1_mean"], tcn["motion_shuffled"]["ranking_f1_mean"]], width, label="TinyTCN", color="#0f766e")
    axes[0].set(xticks=x, xticklabels=["Clean", "Time shuffled"], ylabel="Ranking F1", ylim=(0, 0.75), title="Temporal-order intervention")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar([0, 1], [args.rf_prefall, tcn["prefall"]["roc_auc"]], color=["#64748b", "#0f766e"])
    low, high = tcn["prefall"]["stream_bootstrap_ci_95"]
    axes[1].errorbar(1, tcn["prefall"]["roc_auc"], yerr=[[tcn["prefall"]["roc_auc"] - low], [high - tcn["prefall"]["roc_auc"]]], fmt="none", color="black", capsize=4)
    axes[1].axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1)
    axes[1].set(xticks=[0, 1], xticklabels=["Random Forest", "TinyTCN"], ylabel="Pre-fall ROC-AUC", ylim=(0.45, 1.0), title="Pre-event shortcut predictability")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
