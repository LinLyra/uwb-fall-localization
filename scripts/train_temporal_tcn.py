#!/usr/bin/env python3
"""Train a small TCN and run clean, shuffled, and pre-fall controls."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from audit_prefall import make_examples, read_events
from train_tabular_baselines import decode, load_cache, write_predictions
from uwb_fall.temporal import shuffle_temporal_channels


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        pad = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TinyTCN(nn.Module):
    def __init__(self, inputs: int, width: int = 48) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(inputs, width, 1),
            nn.GELU(),
            TemporalBlock(width, 1),
            TemporalBlock(width, 2),
            TemporalBlock(width, 4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize(train_x: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = np.mean(train_x, axis=(0, 2), keepdims=True)
    std = np.std(train_x, axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-5)
    return tuple(((item - mean) / std).astype(np.float32) for item in (train_x, *others))


def fit_model(x: np.ndarray, y: np.ndarray, seed: int, epochs: int, device: torch.device) -> TinyTCN:
    seed_everything(seed)
    model = TinyTCN(x.shape[1]).to(device)
    positives = max(int(y.sum()), 1)
    weight = torch.tensor([(len(y) - positives) / positives], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_y)
        print(f"epoch {epoch + 1:02d}/{epochs}: loss={total / len(dataset):.5f}", flush=True)
    return model


def predict(model: TinyTCN, x: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    result: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            logits = model(torch.from_numpy(x[start : start + 1024]).to(device))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def nested_events(scores: np.ndarray, val: dict[str, np.ndarray], package: Path) -> tuple[list[dict], dict]:
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    durations = stream_durations(package / "streams" / "val")
    ground = load_ground_truth(package / "streams_events.csv", ignore_flagged=True)
    combined: list[dict] = []
    folds: dict[str, dict] = {}
    for held_out in sorted(durations):
        selection = val["stream_ids"] != held_out
        held = ~selection
        selection_durations = {key: value for key, value in durations.items() if key != held_out}
        best = None
        for threshold in np.arange(0.25, 0.91, 0.05):
            for merge_gap in (0.0, 0.5, 1.0):
                for trim in (0.0, 0.25, 0.5, 0.75):
                    rows = decode(scores[selection], val["stream_ids"][selection], val["starts"][selection], val["ends"][selection], float(threshold), merge_gap, trim)
                    rank = float(evaluate(ground, rows, selection_durations)["ranking_score"]["f1_mean"])
                    if best is None or rank > best[0]:
                        best = (rank, {"threshold": round(float(threshold), 2), "merge_gap": merge_gap, "trim": trim})
        params = best[1]
        rows = decode(scores[held], val["stream_ids"][held], val["starts"][held], val["ends"][held], params["threshold"], params["merge_gap"], params["trim"])
        combined.extend(rows)
        folds[held_out] = {"selected_decoder": params, "selection_ranking_f1": best[0]}
    report = evaluate(ground, combined, durations)
    return combined, {"report": report, "folds": folds}


def frozen_events(
    scores: np.ndarray, val: dict[str, np.ndarray], package: Path, folds: dict[str, dict]
) -> tuple[list[dict], dict]:
    from abc2027.evaluate import evaluate, load_ground_truth, stream_durations

    durations = stream_durations(package / "streams" / "val")
    ground = load_ground_truth(package / "streams_events.csv", ignore_flagged=True)
    rows: list[dict] = []
    for stream_id in sorted(durations):
        mask = val["stream_ids"] == stream_id
        params = folds[stream_id]["selected_decoder"]
        rows.extend(decode(scores[mask], val["stream_ids"][mask], val["starts"][mask], val["ends"][mask], params["threshold"], params["merge_gap"], params["trim"]))
    return rows, {"report": evaluate(ground, rows, durations), "folds": folds}


def summarize(result: dict) -> dict:
    report = result["report"]
    return {
        "ranking_f1_mean": report["ranking_score"]["f1_mean"],
        "f1_by_tiou": {key: value["f1"] for key, value in report["by_theta"].items()},
        "false_alarms_per_hour_at_0.3": report["by_theta"]["0.3"]["false_alarms_per_hour"],
        "folds": result["folds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package / "src"))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device", device, flush=True)
    train = load_cache(args.cache / "train_temporal_w100_h25.npz")
    val = load_cache(args.cache / "val_temporal_w100_h25.npz")
    train_x, val_x = normalize(train["features"], val["features"])
    model = fit_model(train_x, train["labels"], args.seed, args.epochs, device)
    clean_scores = predict(model, val_x, device)
    shuffled_scores = predict(model, shuffle_temporal_channels(val_x, args.seed), device)
    clean_rows, clean_result = nested_events(clean_scores, val, args.package)
    shuffled_rows, shuffled_result = frozen_events(
        shuffled_scores, val, args.package, clean_result["folds"]
    )

    rng = np.random.default_rng(args.seed)
    train_events = read_events(args.package / "streams_events.csv", "train", ignore_flagged=False)
    val_events = read_events(args.package / "streams_events.csv", "val", ignore_flagged=True)
    pre_train_x, pre_train_y, _ = make_examples(train, train_events, rng)
    pre_val_x, pre_val_y, pre_groups = make_examples(val, val_events, rng)
    pre_train_x, pre_val_x = normalize(pre_train_x, pre_val_x)
    pre_model = fit_model(pre_train_x, pre_train_y, args.seed, args.epochs, device)
    pre_scores = predict(pre_model, pre_val_x, device)
    pre_auc = float(roc_auc_score(pre_val_y, pre_scores))
    streams = sorted(set(pre_groups.tolist()))
    per_stream = {stream: float(roc_auc_score(pre_val_y[pre_groups == stream], pre_scores[pre_groups == stream])) for stream in streams}
    bootstrap = []
    for _ in range(10_000):
        sampled = rng.choice(streams, size=len(streams), replace=True)
        chosen = np.concatenate([np.flatnonzero(pre_groups == stream) for stream in sampled])
        bootstrap.append(float(roc_auc_score(pre_val_y[chosen], pre_scores[chosen])))

    args.output.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output / "clean_predictions.csv", clean_rows)
    write_predictions(args.output / "motion_shuffled_predictions.csv", shuffled_rows)
    torch.save({"state_dict": model.state_dict(), "seed": args.seed, "epochs": args.epochs}, args.output / "tiny_tcn.pt")
    summary = {
        "model": "TinyTCN: 1x1 projection, three residual dilated Conv1d blocks, global pooling",
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "parameters": sum(item.numel() for item in model.parameters()),
        "clean": summarize(clean_result),
        "motion_shuffled": {**summarize(shuffled_result), "decoder": "clean nested decoder frozen"},
        "prefall": {
            "roc_auc": pre_auc,
            "stream_bootstrap_ci_95": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
            "per_stream_roc_auc": per_stream,
            "validation_examples": len(pre_val_y),
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
