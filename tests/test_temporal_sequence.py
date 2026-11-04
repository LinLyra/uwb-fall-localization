import numpy as np

from uwb_fall.temporal import shuffle_temporal_channels, temporal_power_sequence


def test_temporal_sequence_shape_and_radar_invariance() -> None:
    rng = np.random.default_rng(7)
    cir = rng.normal(size=(3, 100, 64)).astype(np.float32)
    expected = temporal_power_sequence(cir, range_bands=8)
    actual = temporal_power_sequence(cir[[2, 0, 1]], range_bands=8)
    assert expected.shape == (24, 25)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_shuffle_preserves_channel_values_but_changes_order() -> None:
    values = np.arange(2 * 4 * 25).reshape(2, 4, 25)
    shuffled = shuffle_temporal_channels(values, seed=3)
    np.testing.assert_array_equal(np.sort(shuffled, axis=2), np.sort(values, axis=2))
    assert not np.array_equal(shuffled, values)
