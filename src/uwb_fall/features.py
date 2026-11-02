"""Leakage-resistant features for blockwise-I/Q UWB CIR windows."""

from __future__ import annotations

import numpy as np


def complex_magnitude(cir: np.ndarray) -> np.ndarray:
    """Convert ``[..., I(0:B), Q(0:B)]`` CIR samples to magnitude."""
    values = np.asarray(cir)
    if values.shape[-1] % 2:
        raise ValueError("the CIR feature dimension must contain equal I/Q halves")
    bins = values.shape[-1] // 2
    i = values[..., :bins].astype(np.float64, copy=False)
    q = values[..., bins:].astype(np.float64, copy=False)
    return np.hypot(i, q)


def _summarize(series: np.ndarray, axis: int) -> np.ndarray:
    return np.concatenate(
        [
            np.mean(series, axis=axis),
            np.std(series, axis=axis),
            np.max(series, axis=axis),
            np.quantile(series, 0.90, axis=axis),
        ]
    )


def window_features_v2(cir: np.ndarray, range_bands: int = 16) -> np.ndarray:
    """Extract physics-aware motion features from one ``(R,T,2B)`` window.

    Features combine temporal dynamics, coarse range occupancy, cross-radar
    energy fractions, and peak-range motion. Per-window centering suppresses
    the static-room profile and reduces the documented pre-fall shortcut.
    """
    values = np.asarray(cir)
    if values.ndim != 3:
        raise ValueError("expected CIR with shape (radars, time, iq_features)")
    if values.shape[1] < 2:
        raise ValueError("at least two time samples are required")
    if range_bands < 1:
        raise ValueError("range_bands must be positive")

    bins = values.shape[2] // 2
    i = values[:, :, :bins].astype(np.float64, copy=False)
    q = values[:, :, bins:].astype(np.float64, copy=False)
    i = i - np.mean(i, axis=1, keepdims=True)
    q = q - np.mean(q, axis=1, keepdims=True)
    # Center complex components before magnitude. Centering magnitude itself is
    # not invariant to a static complex offset and would retain room identity.
    power = i * i + q * q

    return power_features(power, range_bands=range_bands)


def power_features(power: np.ndarray, range_bands: int = 16) -> np.ndarray:
    """Extract the v2 feature vector from complex residual power ``(R,T,B)``."""
    power = np.asarray(power, dtype=np.float64)
    if power.ndim != 3 or power.shape[1] < 2:
        raise ValueError("expected residual power with shape (radars, time, range_bins)")
    if range_bands < 1:
        raise ValueError("range_bands must be positive")

    frame_energy = np.mean(power, axis=2)
    temporal = _summarize(frame_energy, axis=1)
    velocity = _summarize(np.abs(np.diff(frame_energy, axis=1)), axis=1)

    band_features: list[np.ndarray] = []
    for indices in np.array_split(np.arange(power.shape[2]), range_bands):
        band = np.mean(power[:, :, indices], axis=(1, 2))
        band_features.append(np.log1p(band))
    ranges = np.concatenate(band_features)

    radar_energy = np.mean(power, axis=(1, 2))
    fractions = radar_energy / max(float(np.sum(radar_energy)), np.finfo(float).eps)
    entropy = -float(np.sum(fractions * np.log(fractions + np.finfo(float).eps)))

    peak_bins = np.argmax(power, axis=2).astype(np.float64)
    peak_motion = np.concatenate(
        [np.std(peak_bins, axis=1), np.mean(np.abs(np.diff(peak_bins, axis=1)), axis=1)]
    )

    result = np.concatenate([temporal, velocity, ranges, fractions, [entropy], peak_motion])
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def shift_range(power: np.ndarray, offset: int) -> np.ndarray:
    """Translate range bins with zero fill and no circular wraparound."""
    values = np.asarray(power)
    shifted = np.zeros_like(values)
    if offset > 0:
        shifted[..., offset:] = values[..., :-offset]
    elif offset < 0:
        shifted[..., :offset] = values[..., -offset:]
    else:
        shifted[...] = values
    return shifted
