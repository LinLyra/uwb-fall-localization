import numpy as np

from scripts.analyze_boundary_limits import allowed_error, temporal_iou


def test_translation_error_formula_hits_requested_tiou() -> None:
    duration = 2.0
    for threshold in (0.3, 0.5, 0.7):
        error = float(allowed_error(np.asarray([duration]), threshold, "translation")[0])
        assert abs(temporal_iou(0.0, duration, error, duration + error) - threshold) < 1e-12


def test_inward_and_outward_formulas() -> None:
    duration = np.asarray([2.0])
    assert np.isclose(allowed_error(duration, 0.5, "inward")[0], 0.5)
    assert np.isclose(allowed_error(duration, 0.5, "outward")[0], 1.0)
