import numpy as np

from uwb_fall.features import shift_range


def test_range_shift_zero_fills_without_wraparound() -> None:
    power = np.arange(5, dtype=float).reshape(1, 1, 5)
    np.testing.assert_array_equal(shift_range(power, 2), [[[0, 0, 0, 1, 2]]])
    np.testing.assert_array_equal(shift_range(power, -2), [[[2, 3, 4, 0, 0]]])
    np.testing.assert_array_equal(shift_range(power, 0), power)
