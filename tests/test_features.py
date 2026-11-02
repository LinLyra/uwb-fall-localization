import numpy as np
import pytest

from uwb_fall.features import complex_magnitude, power_features, window_features_v2


def test_blockwise_iq_conversion() -> None:
    cir = np.array([[[3.0, 5.0, 4.0, 12.0]]])
    np.testing.assert_allclose(complex_magnitude(cir), [[[5.0, 13.0]]])


def test_features_are_finite_and_static_offset_invariant() -> None:
    rng = np.random.default_rng(2027)
    cir = rng.normal(size=(3, 20, 12)).astype(np.float32)
    offset = np.concatenate([np.full(6, 7.0), np.full(6, -2.0)])
    first = window_features_v2(cir, range_bands=3)
    second = window_features_v2(cir + offset, range_bands=3)
    assert np.all(np.isfinite(first))
    np.testing.assert_allclose(first, second, rtol=1e-5, atol=1e-5)


def test_invalid_shapes_are_rejected() -> None:
    with pytest.raises(ValueError):
        complex_magnitude(np.zeros((2, 3, 5)))
    with pytest.raises(ValueError):
        window_features_v2(np.zeros((2, 8)))
    with pytest.raises(ValueError):
        power_features(np.zeros((2, 8)))
