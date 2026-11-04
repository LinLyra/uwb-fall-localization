"""Temporal decoding, evaluation, and order-sensitive representations."""

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


@dataclass(frozen=True)
class ScoredEvent:
    """Predicted event with a confidence used by official greedy matching."""

    event: Event
    score: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")


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
    predictions: Iterable[ScoredEvent], targets: Iterable[Event], *, iou_threshold: float = 0.3
) -> dict[str, float | int]:
    """Match predictions one-to-one in descending confidence order."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must lie in [0, 1]")
    predicted = sorted(predictions, key=lambda item: item.score, reverse=True)
    truth = list(targets)
    used_t: set[int] = set()
    matches: list[tuple[Event, Event, float]] = []
    for prediction in predicted:
        candidates = [(temporal_iou(prediction.event, target), index) for index, target in enumerate(truth) if index not in used_t]
        if not candidates:
            continue
        iou, index = max(candidates)
        if iou >= iou_threshold:
            used_t.add(index)
            matches.append((prediction.event, truth[index], iou))
    hits = len(matches)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "hits": hits,
        "misses": len(truth) - hits,
        "false_alarms": len(predicted) - hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_tiou": float(np.mean([item[2] for item in matches])) if matches else 0.0,
        "mean_start_error_samples": float(np.mean([abs(p.start - t.start) for p, t, _ in matches])) if matches else 0.0,
        "mean_end_error_samples": float(np.mean([abs(p.end - t.end) for p, t, _ in matches])) if matches else 0.0,
    }


def temporal_power_sequence(
    cir: np.ndarray, *, time_steps: int = 25, range_bands: int = 16
) -> np.ndarray:
    """Return radar-order invariant residual power as ``(3*bands, time)``."""
    values = np.asarray(cir)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("expected CIR with shape (3, time, iq_features)")
    if values.shape[1] % time_steps:
        raise ValueError("time samples must be divisible by time_steps")
    bins = values.shape[2] // 2
    if bins * 2 != values.shape[2]:
        raise ValueError("expected equal I/Q range-bin halves")
    i = values[:, :, :bins].astype(np.float32, copy=False)
    q = values[:, :, bins:].astype(np.float32, copy=False)
    i = i - np.mean(i, axis=1, keepdims=True)
    q = q - np.mean(q, axis=1, keepdims=True)
    power = i * i + q * q
    power = power.reshape(3, time_steps, values.shape[1] // time_steps, bins).mean(axis=2)
    reduced = np.stack(
        [np.mean(power[..., indices], axis=-1) for indices in np.array_split(np.arange(bins), range_bands)],
        axis=-1,
    )
    reduced = np.sort(reduced, axis=0)
    sequence = np.log1p(reduced).transpose(0, 2, 1).reshape(3 * range_bands, time_steps)
    return np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def shuffle_temporal_channels(sequences: np.ndarray, seed: int) -> np.ndarray:
    """Destroy within-channel temporal order with one fixed permutation map."""
    values = np.asarray(sequences)
    if values.ndim != 3:
        raise ValueError("expected sequences with shape (windows, channels, time)")
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random(values.shape[1:]), axis=1)
    return np.take_along_axis(values, order[None, :, :], axis=2)
