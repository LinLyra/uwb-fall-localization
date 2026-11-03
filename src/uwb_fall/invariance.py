"""Radar-order canonicalization for the 82-dimensional feature layout."""

from __future__ import annotations

import numpy as np


RADAR_GROUP_STARTS = tuple(
    list(range(0, 12, 3))
    + list(range(12, 24, 3))
    + list(range(24, 72, 3))
    + [72, 76, 79]
)


def canonicalize_radar_features(features: np.ndarray) -> np.ndarray:
    """Sort every three-radar feature group while preserving scalar entropy.

    Sorting is a deterministic set representation. It removes radar identity
    but retains the low/middle/high response for each statistic or range band.
    """
    values = np.asarray(features)
    if values.shape[-1] != 82:
        raise ValueError("expected the 82-dimensional v2 feature layout")
    result = values.copy()
    for start in RADAR_GROUP_STARTS:
        result[..., start : start + 3] = np.sort(result[..., start : start + 3], axis=-1)
    return result
