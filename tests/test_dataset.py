from pathlib import Path

import numpy as np

from uwb_fall.dataset import WindowSet, save_window_set, temporal_iou, window_label


def test_temporal_iou_and_window_label() -> None:
    assert temporal_iou(0.0, 2.0, 1.0, 3.0) == 1.0 / 3.0
    assert window_label(0.0, 2.0, [(1.0, 3.0)], minimum_iou=0.3) == 1
    assert window_label(0.0, 1.0, [(2.0, 3.0)]) == 0


def test_cache_round_trip(tmp_path: Path) -> None:
    windows = WindowSet(
        features=np.ones((2, 3), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.uint8),
        stream_ids=np.array(["train_00", "train_00"]),
        starts=np.array([0.0, 1.0], dtype=np.float32),
        ends=np.array([2.0, 3.0], dtype=np.float32),
    )
    target = tmp_path / "cache.npz"
    save_window_set(target, windows)
    with np.load(target) as loaded:
        np.testing.assert_array_equal(loaded["features"], windows.features)
        np.testing.assert_array_equal(loaded["labels"], windows.labels)
