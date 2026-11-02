"""Deterministic temporal decoding and event-level evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class Event:
    """Half-open interval ``[start, end)`` in stream samples."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("event must satisfy 0 <= start < end")


def temporal_iou(a: Event, b: Event) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return intersection / union if union else 0.0


def hysteresis_decode(
    probabilities: Sequence[float],
    *,
    start_threshold: float,
    end_threshold: float,
    min_duration: int = 1,
    merge_gap: int = 0,
) -> list[Event]:
    """Decode frame probabilities using separate enter and exit thresholds."""
    if not 0 <= end_threshold <= start_threshold <= 1:
        raise ValueError("require 0 <= end_threshold <= start_threshold <= 1")
    if min_duration < 1 or merge_gap < 0:
        raise ValueError("invalid duration or gap")

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("probabilities must be a finite one-dimensional sequence")

    raw: list[Event] = []
    start: int | None = None
    for index, probability in enumerate(values):
        if start is None and probability >= start_threshold:
            start = index
        elif start is not None and probability < end_threshold:
            if index - start >= min_duration:
                raw.append(Event(start, index))
            start = None
    if start is not None and len(values) - start >= min_duration:
        raw.append(Event(start, len(values)))

    merged: list[Event] = []
    for event in raw:
        if merged and event.start - merged[-1].end <= merge_gap:
            merged[-1] = Event(merged[-1].start, event.end)
        else:
            merged.append(event)
    return merged


def event_metrics(
    predictions: Iterable[Event],
    targets: Iterable[Event],
    *,
    iou_threshold: float = 0.3,
) -> dict[str, float | int]:
    """Greedily match events by descending tIoU and report event statistics."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in [0, 1]")
    predicted = list(predictions)
    truth = list(targets)
    candidates = sorted(
        (
            (temporal_iou(p, t), pi, ti)
            for pi, p in enumerate(predicted)
            for ti, t in enumerate(truth)
        ),
        reverse=True,
    )
    used_p: set[int] = set()
    used_t: set[int] = set()
    matches: list[tuple[Event, Event, float]] = []
    for iou, pi, ti in candidates:
        if iou < iou_threshold or pi in used_p or ti in used_t:
            continue
        used_p.add(pi)
        used_t.add(ti)
        matches.append((predicted[pi], truth[ti], iou))

    hits = len(matches)
    false_alarms = len(predicted) - hits
    misses = len(truth) - hits
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_tiou": float(np.mean([m[2] for m in matches])) if matches else 0.0,
        "mean_start_error_samples": float(
            np.mean([abs(p.start - t.start) for p, t, _ in matches])
        ) if matches else 0.0,
        "mean_end_error_samples": float(
            np.mean([abs(p.end - t.end) for p, t, _ in matches])
        ) if matches else 0.0,
    }

