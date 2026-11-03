import itertools

import numpy as np
import pytest

from uwb_fall.features import power_features
from uwb_fall.invariance import canonicalize_radar_features


def test_canonical_features_are_radar_permutation_invariant() -> None:
    rng = np.random.default_rng(2027)
    power = rng.random((3, 30, 184))
    reference = canonicalize_radar_features(power_features(power))
    for order in itertools.permutations(range(3)):
        candidate = canonicalize_radar_features(power_features(power[list(order)]))
        np.testing.assert_allclose(candidate, reference, rtol=0.0, atol=0.0)


def test_canonicalizer_checks_layout() -> None:
    with pytest.raises(ValueError):
        canonicalize_radar_features(np.zeros(81))
