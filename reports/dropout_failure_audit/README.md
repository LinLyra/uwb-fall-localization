# Missing-radar evaluation

This comparison uses the same radar-order invariant representation. The only
difference is whether 25% radar-dropout augmentation was added during training.
Models and nested decoder settings are frozen before each failure condition.

| Condition | Clean-trained RF | Dropout-trained RF | Fraction of dropout model's clean score retained |
|---|---:|---:|---:|
| All radars | 0.6619 | 0.6462 | 100% |
| Radar 0 missing | 0.0000 | 0.6450 | 99.8% |
| Radar 1 missing | 0.0000 | 0.4485 | 69.4% |
| Radar 2 missing | 0.0000 | 0.3467 | 53.6% |
| Highest-energy radar missing per window | 0.0000 | 0.0444 | 6.9% |

Radar dropout succeeds at its stated target. The clean-trained model emits no
usable events after any fixed radar is zeroed. Because zero filling is an
out-of-distribution representation of sensor failure, the
[fair imputation audit](../imputation_audit/README.md) also compares against
replacing the failed sensor with the mean of the two available sensors. Dropout
training remains clearly better for radar 0 and radar 2; its advantage for radar
1 is small. Radar 0 can be removed with almost no loss, while radar 2 remains the
weakest failure case.

The method does not handle adversarial removal of the strongest radar in each
window. That condition changes which physical sensor is absent over time and
removes the dominant evidence by construction. It should not be summarized as
ordinary single-sensor failure.

Dropout training costs 0.0158 clean ranking F1, with a paired 95% interval that
includes zero, and reduces clean false alarms from 11.23 to 9.76 per hour. Given
the large fixed-failure recovery, radar dropout is retained as a robustness
component even though it is not a clean-score improvement.

All five non-identity radar orders reproduce the clean dropout score exactly
(0.6462), confirming that ordering invariance comes from the representation and
missing-sensor robustness comes from training augmentation.
