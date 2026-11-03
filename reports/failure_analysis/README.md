# Event-level failure analysis

Failures are matched at tIoU 0.3 using the official evaluator and the nested
82-feature Random Forest predictions. Flagged validation events are ignored.

## Counts by stream

| Stream | TP | FP | FN | Predictions ignored near flagged events |
|---|---:|---:|---:|---:|
| val_00 | 17 | 8 | 4 | 11 |
| val_01 | 15 | 3 | 3 | 7 |
| val_02 | 13 | 9 | 6 | 11 |
| val_03 | 3 | 6 | 2 | 7 |
| val_04 | 3 | 1 | 1 | 1 |

`val_03` is difficult for two separate reasons: it has more false positives
than true positives, and two of five evaluable falls are missed at the official
threshold.

## val_03 false positives

- Three predictions overlap source segments labelled `other:Autre`.
- One overlaps the daily activity `Marcher` (walking).
- Two occur inside fall-source segments but fail to match the annotated event.

The prediction at 1450.25–1450.75 seconds overlaps only the tail of the
1449.24–1450.38 event and falls below tIoU 0.3. It is a boundary near-miss, not
an unrelated activity detection. The prediction at 1569.75–1575.25 seconds ends
just before the 1575.28–1576.92 event starts. This is consistent with the strong
pre-fall shortcut found in the separate control.

## val_03 matched-event boundary errors

The three true positives have tIoU 0.630, 0.369, and 0.627. One prediction
starts 1.39 seconds late; the other two extend 0.71–1.15 seconds past the
annotated end. The low stream score is therefore not explained by classification
alone. Late activation, early pre-fall activation, and overlong event decoding
all occur in the same stream.

## Consequence

A single larger classifier is unlikely to address these failures. The next
method needs separate onset and offset evidence, while shortcut mitigation must
prevent pre-fall context from firing an event early. The source labels in this
report describe the synthetic stream composition and are used only for failure
analysis, not as model inputs.
