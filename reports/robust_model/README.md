# Robustness augmentation v1

The radar-order invariant training set is doubled. Every clean window is kept
and receives one augmented copy. Alternating copies use either a random range
shift from {-16, -8, +8, +16} bins or one randomly removed radar; all copies
also receive a power scale from 0.7 to 1.3. Seed 2027 fixes the transforms.

## Clean nested result

| Training | Ranking F1 | F1@0.3 | F1@0.7 | False alarms/hour @0.3 |
|---|---:|---:|---:|---:|
| Invariant RF, clean only | 0.6619 | 0.7234 | 0.1560 | 11.23 |
| Invariant RF, augmentation v1 | 0.6309 | 0.6846 | 0.1342 | 15.13 |

The paired ranking-F1 change is -0.0311 with a stream-bootstrap interval of
[-0.0896, 0.0064]. The interval includes zero, but the point estimate and false
alarm rate both move in the wrong direction. `val_01` accounts for the largest
drop (-0.1113).

## Decision

This augmentation recipe is not promoted. Doubling the data with a strong
perturbation on every added window likely overweights inputs far from the clean
distribution. The result is retained as a negative experiment.

The next version should separate the interventions and reduce their sampling
rate. Range shifts and radar dropout must be evaluated independently before
combining them. Clean F1 and the corresponding spatial control score should be
reported as a pair; a clean-performance loss is acceptable only if robustness
improves enough to justify it.
