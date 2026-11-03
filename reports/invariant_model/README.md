# Radar-order invariant baseline

Every feature associated with the three radars is represented as a sorted
three-value group. The model therefore sees the low, middle, and high response
for each statistic or range band, but not the radar serial position. Scalar
cross-radar entropy is unchanged.

## Clean nested evaluation

| Representation | Ranking F1 | F1@0.3 | F1@0.7 | False alarms/hour @0.3 |
|---|---:|---:|---:|---:|
| Ordered radar features | 0.6575 | 0.7034 | 0.1931 | 13.18 |
| Canonically sorted radar features | 0.6619 | 0.7234 | 0.1560 | 11.23 |

The paired ranking-F1 difference is +0.0045 with a stream-bootstrap interval of
[-0.0249, 0.0572] and an exact randomization p-value of 1.0. There is no
evidence of a clean-performance penalty or improvement.

Unit tests enumerate all six radar permutations and verify exact equality after
canonicalization. Radar reordering is therefore solved by construction rather
than inferred from a small validation sample.

## What remains unsolved

Pre-fall ROC-AUC remains 0.9403, essentially unchanged from 0.9407 for the
ordered representation. Canonicalization removes radar identity but does not
remove context or fixed-range information. It also assumes that all three
radars are present; it does not provide missing-sensor robustness.

This method is retained as a simpler and safer representation, not as a claimed
accuracy improvement. The next experiments require training-time range shifts
and radar dropout, followed by the same clean and shortcut-control evaluations.
