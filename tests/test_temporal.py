import pytest

from uwb_fall.temporal import Event, event_metrics, hysteresis_decode, temporal_iou


def test_temporal_iou() -> None:
    assert temporal_iou(Event(10, 20), Event(15, 25)) == pytest.approx(1 / 3)


def test_hysteresis_merges_short_gap() -> None:
    probabilities = [0.0, 0.8, 0.8, 0.2, 0.0, 0.9, 0.9, 0.2]
    events = hysteresis_decode(
        probabilities,
        start_threshold=0.7,
        end_threshold=0.3,
        min_duration=2,
        merge_gap=2,
    )
    assert events == [Event(1, 7)]


def test_event_metrics_matches_once() -> None:
    report = event_metrics([Event(9, 21), Event(40, 45)], [Event(10, 20)])
    assert report["hits"] == 1
    assert report["false_alarms"] == 1
    assert report["misses"] == 0

