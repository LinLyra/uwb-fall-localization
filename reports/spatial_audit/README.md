# Spatial shortcut audit

The clean 82-feature Random Forest is evaluated after deterministic input
transformations. Training weights and leave-one-stream-out decoder settings are
unchanged. The clean reference ranking F1 is 0.6575.

## Range translation

| Range-bin shift | Ranking F1 |
|---:|---:|
| -24 | 0.0000 |
| -16 | 0.1870 |
| -8 | 0.6612 |
| 0 | 0.6575 |
| +8 | 0.3365 |
| +16 | 0.0000 |
| +24 | 0.0000 |

The response is strongly asymmetric. A shift of -8 bins has no measurable
cost: the paired difference is -0.0037 with a 95% interval of
[-0.0750, 0.0667]. A +8-bin shift loses 0.3210, with a paired interval of
[0.1774, 0.4399]. The median per-stream relative loss for +8 bins is about 53%.

A random contiguous mask covering 12.5% of range bins loses 0.0437 overall,
with a paired interval of [-0.0334, 0.1190]. Its median per-stream relative loss
is about 3%. The +8 translation therefore exceeds the preregistered 25% loss
threshold without a comparable loss under random masking. H3 is rejected: the
model is sensitive to absolute range position, not just the presence of motion
across a coarse range pattern.

## Range masking

| Masked range fraction | Ranking F1 |
|---:|---:|
| 12.5% | 0.6138 |
| 25% | 0.1758 |
| 50% | 0.0000 |

The sharp nonlinear decline suggests that evidence is concentrated rather than
evenly distributed across the 184 range bins. The random mask location is fixed
by seed 2027 for reproducibility; repeated-mask seeds are still needed before
making a precise localization claim.

## Radar removal and ordering

| Control | Ranking F1 |
|---|---:|
| Drop radar 0 | 0.0000 |
| Drop radar 1 | 0.0422 |
| Drop radar 2 | 0.0000 |
| Drop highest-energy radar per window | 0.0000 |
| Best non-identity radar order (`021`) | 0.3061 |
| Other non-identity orders | 0.0000–0.0433 |

H4 is rejected. The representation is neither robust to one missing radar nor
invariant to radar ordering. Zeroing a radar is intentionally out of the model's
training distribution and changes cross-radar fractions, so this experiment
does not show that the removed sensor alone contains all information. It shows
that the current pipeline has no graceful sensor-failure behavior.

## Implication

The next model should not retain radar identity and absolute range position in
their current form. Candidate corrections are range-shift augmentation,
motion-centred alignment, radar dropout during training, and a permutation-
equivariant radar encoder. Each correction must reduce the control sensitivity
under nested evaluation; clean validation F1 alone is no longer sufficient.
