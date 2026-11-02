"""Chunk-safe feature extraction from official continuous HDF5 streams."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .features import window_features_v2


@dataclass(frozen=True)
class WindowSet:
    features: np.ndarray
    labels: np.ndarray
    stream_ids: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def temporal_iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    intersection = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    union = max(a_end, b_end) - min(a_start, b_start)
    return intersection / union if union > 0 else 0.0


def read_events(path: Path, split: str) -> dict[str, list[tuple[float, float]]]:
    result: dict[str, list[tuple[float, float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stream_id = row["stream_id"]
            if not stream_id.startswith(f"{split}_"):
                continue
            result.setdefault(stream_id, []).append((float(row["t_start"]), float(row["t_end"])))
    return result


def window_label(
    start: float,
    end: float,
    events: list[tuple[float, float]],
    minimum_iou: float = 0.1,
) -> int:
    return int(any(temporal_iou(start, end, a, b) >= minimum_iou for a, b in events))


def extract_split(
    streams_dir: Path,
    events_csv: Path,
    split: str,
    *,
    window: int = 100,
    hop: int = 25,
    rate_hz: float = 50.0,
    range_bands: int = 16,
) -> WindowSet:
    """Extract a split without ever loading an entire stream into memory."""
    if window <= 1 or hop <= 0:
        raise ValueError("window must exceed one sample and hop must be positive")
    events = read_events(events_csv, split)
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_streams: list[str] = []
    all_starts: list[float] = []
    all_ends: list[float] = []

    paths = sorted((streams_dir / split).glob("stream_*.h5"))
    if not paths:
        raise FileNotFoundError(f"no HDF5 streams under {streams_dir / split}")
    for path in paths:
        stream_id = f"{split}_{path.stem.rsplit('_', 1)[-1]}"
        stream_events = events.get(stream_id, [])
        with h5py.File(path, "r") as handle:
            cir = handle["cir"]
            count = int(cir.shape[1])
            for index in range(0, count - window + 1, hop):
                start = index / rate_hz
                end = (index + window) / rate_hz
                block = cir[:, index : index + window, :]
                all_x.append(window_features_v2(block, range_bands=range_bands))
                all_y.append(window_label(start, end, stream_events))
                all_streams.append(stream_id)
                all_starts.append(start)
                all_ends.append(end)

    return WindowSet(
        features=np.stack(all_x),
        labels=np.asarray(all_y, dtype=np.uint8),
        stream_ids=np.asarray(all_streams),
        starts=np.asarray(all_starts, dtype=np.float32),
        ends=np.asarray(all_ends, dtype=np.float32),
    )


def save_window_set(path: Path, windows: WindowSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=windows.features,
        labels=windows.labels,
        stream_ids=windows.stream_ids,
        starts=windows.starts,
        ends=windows.ends,
    )
