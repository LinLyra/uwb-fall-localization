# Boundary and tIoU sensitivity

This analysis separates model boundary error from the metric sensitivity caused
by short events and uncertain timestamps. It uses the 67 evaluable validation
events. It is not an annotation-agreement study because no independent second
annotation is available.

## Event duration

| Statistic | Duration (s) |
|---|---:|
| Minimum | 1.02 |
| 25th percentile | 1.53 |
| Median | 2.00 |
| 75th percentile | 2.24 |
| 90th percentile | 2.85 |
| Maximum | 4.40 |

For a translated prediction of the same duration, an absolute shift `e` gives

```text
tIoU = (D - e) / (D + e)
```

for `0 ≤ e < D`. A median two-second event can move by at most 0.353 seconds
and still reach tIoU 0.7. If both boundaries move inward instead, the allowable
symmetric error is only 0.300 seconds. High-tIoU evaluation is therefore
sensitive to a few hundred milliseconds.

## Timestamp-uncertainty simulation

Start and end boundaries are independently perturbed from a uniform interval.
Ten thousand replicates use seed 2027.

| Perturbation | Fraction ≥0.3 | Fraction ≥0.5 | Fraction ≥0.7 |
|---|---:|---:|---:|
| ±0.5 s | 0.998 | 0.973 | 0.739 |
| ±1.0 s | 0.899 | 0.698 | 0.283 |

The current nested RF reaches F1 0.193 at tIoU 0.7. Under the stronger ±1 s
simulation, even otherwise perfect boundaries exceed 0.7 only about 28% of the
time. Under ±0.5 s noise, the simulated fraction remains about 74%. The
documented ±0.5–1 s uncertainty therefore could explain a substantial part of
the high-tIoU ceiling, but the available documentation is not enough to choose
one noise model or claim that 0.283 is the true ceiling.

## Decision

F1 at tIoU 0.7 remains useful as a sensitivity measure, but it should not be the
sole optimization target. A boundary model should report start/end absolute
error, F1 across all official thresholds, and performance after tolerance-aware
boundary analysis. Independent re-annotation would be required before making a
strong claim about the attainable ceiling.
