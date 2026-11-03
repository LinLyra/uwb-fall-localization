#!/usr/bin/env python3
"""Produce per-stream TP, FP, FN, and boundary-error records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def read_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {
                **row,
                "t_start": float(row["t_start"]),
                "t_end": float(row["t_end"]),
                "score": float(row["score"]),
                "t_alarm": float(row["t_alarm"]) if row.get("t_alarm") else None,
            }
            for row in csv.DictReader(handle)
        ]


def read_composition(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            {**row, "t_start": float(row["t_start"]), "t_end": float(row["t_end"])}
            for row in csv.DictReader(handle)
        ]


def source_labels(prediction: dict, composition: list[dict]) -> str:
    labels = []
    for segment in composition:
        if segment["stream_id"] != prediction["stream_id"]:
            continue
        if min(prediction["t_end"], segment["t_end"]) > max(prediction["t_start"], segment["t_start"]):
            labels.append(f"{segment['kind']}:{segment['label']}")
    return " | ".join(dict.fromkeys(labels)) or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.3)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    from abc2027.evaluate import IGNORE_MARGIN_S, _match_stream, load_ground_truth, stream_durations

    ground = load_ground_truth(args.package / "streams_events.csv", ignore_flagged=True)
    predictions = read_predictions(args.predictions)
    composition = read_composition(args.package / "streams_composition.csv")
    durations = stream_durations(args.package / "streams" / "val")
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for stream_id in sorted(durations):
        gt = [item for item in ground if item["stream_id"] == stream_id]
        pred = [item for item in predictions if item["stream_id"] == stream_id]
        result = _match_stream(gt, pred, args.theta)
        matched_pred = {pair[0] for pair in result["pairs"]}
        matched_gt = {pair[1] for pair in result["pairs"]}
        scored = result["scored"]
        ignored = [item for item in gt if item["ignore"]]
        for pred_index, gt_index, iou in result["pairs"]:
            prediction = pred[pred_index]
            event = scored[gt_index]
            rows.append(
                {
                    "stream_id": stream_id,
                    "outcome": "TP",
                    "pred_start": prediction["t_start"],
                    "pred_end": prediction["t_end"],
                    "gt_start": event["t_start"],
                    "gt_end": event["t_end"],
                    "tiou": iou,
                    "start_error": prediction["t_start"] - event["t_start"],
                    "end_error": prediction["t_end"] - event["t_end"],
                    "score": prediction["score"],
                    "source_labels": source_labels(prediction, composition),
                }
            )
        for index, prediction in enumerate(pred):
            if index in matched_pred:
                continue
            touches_ignore = any(
                min(prediction["t_end"], event["t_end"] + IGNORE_MARGIN_S)
                > max(prediction["t_start"], event["t_start"] - IGNORE_MARGIN_S)
                for event in ignored
            )
            if touches_ignore:
                continue
            rows.append(
                {
                    "stream_id": stream_id,
                    "outcome": "FP",
                    "pred_start": prediction["t_start"],
                    "pred_end": prediction["t_end"],
                    "gt_start": "",
                    "gt_end": "",
                    "tiou": 0.0,
                    "start_error": "",
                    "end_error": "",
                    "score": prediction["score"],
                    "source_labels": source_labels(prediction, composition),
                }
            )
        for index, event in enumerate(scored):
            if index in matched_gt:
                continue
            rows.append(
                {
                    "stream_id": stream_id,
                    "outcome": "FN",
                    "pred_start": "",
                    "pred_end": "",
                    "gt_start": event["t_start"],
                    "gt_end": event["t_end"],
                    "tiou": 0.0,
                    "start_error": "",
                    "end_error": "",
                    "score": "",
                    "source_labels": "",
                }
            )
        local = [row for row in rows if row["stream_id"] == stream_id]
        summary[stream_id] = {
            "tp": sum(row["outcome"] == "TP" for row in local),
            "fp": sum(row["outcome"] == "FP" for row in local),
            "fn": sum(row["outcome"] == "FN" for row in local),
            "ignored_predictions": result["ignored_predictions"],
        }

    args.output.mkdir(parents=True, exist_ok=True)
    fields = [
        "stream_id", "outcome", "pred_start", "pred_end", "gt_start", "gt_end",
        "tiou", "start_error", "end_error", "score", "source_labels",
    ]
    with (args.output / "failure_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "failure_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    target = "val_03"
    target_gt = [item for item in ground if item["stream_id"] == target and not item["ignore"]]
    target_pred = [item for item in predictions if item["stream_id"] == target]
    fig, ax = plt.subplots(figsize=(12, 3.2), facecolor="white")
    for event in target_gt:
        ax.broken_barh([(event["t_start"], event["t_end"] - event["t_start"])], (1.1, 0.55), facecolors="#0F766E")
    for prediction in target_pred:
        ax.broken_barh(
            [(prediction["t_start"], prediction["t_end"] - prediction["t_start"])],
            (0.2, 0.55),
            facecolors="#38BDF8",
        )
    ax.set_yticks([0.475, 1.375], ["Predictions", "Ground truth"])
    ax.set_xlabel("Time (s)")
    ax.set_title("val_03 event timeline at tIoU 0.3")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.output / "val_03_timeline.png", dpi=220, bbox_inches="tight")
    fig.savefig(args.output / "val_03_timeline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
