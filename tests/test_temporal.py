import pytest

from uwb_fall.temporal import (
    Event,
    ScoredEvent,
    event_metrics,
    hysteresis_decode,
    temporal_iou,
)


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
    report = event_metrics(
        [ScoredEvent(Event(9, 21), 0.9), ScoredEvent(Event(40, 45), 0.2)],
        [Event(10, 20)],
    )
    assert report["hits"] == 1
    assert report["false_alarms"] == 1
    assert report["misses"] == 0


def test_confidence_order_controls_official_greedy_match() -> None:
    target = Event(10, 20)
    predictions = [
        ScoredEvent(Event(10, 16), 0.9),
        ScoredEvent(Event(10, 20), 0.1),
    ]
    report = event_metrics(predictions, [target], iou_threshold=0.3)
    assert report["hits"] == 1
    assert report["false_alarms"] == 1
    assert report["mean_tiou"] == pytest.approx(0.6)

