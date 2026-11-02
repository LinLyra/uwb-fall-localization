# Official protocol audit

Audited against `README_v1.1.md` dated October 2026 from the public ABC 2027
UWB package. The source README remains under `data/raw/` and is not committed.

## Signal schema

- Three radars: serials 29171, 29444 and 30722.
- Common 50 Hz grid.
- HDF5 `cir` shape: `(3, T, 368)`, float32.
- The 368 channels encode 184 complex range bins block-wise as
  `[I_0 ... I_183, Q_0 ... Q_183]`.
- Amplitude for range bin `k` is `sqrt(I_k^2 + Q_k^2)`.

## Official splits

- Train: 13 streams, approximately 6.18 hours.
- Validation: 5 streams, approximately 2.05 hours.
- Test: 5 unlabeled streams, approximately 2.07 hours.
- Train, validation and test use different subjects.
- `segments/` contains train and validation sources; using validation segments
  for training invalidates an unbiased validation estimate.

## Submission and matching

Required columns are `stream_id,t_start,t_end,score`; `t_alarm` is optional.
Predictions are matched one-to-one in descending score order. A match requires
temporal IoU at or above the evaluated threshold.

Primary ranking is the mean event F1 over tIoU thresholds 0.1, 0.3 and 0.5.
Ties are broken by lower median absolute start error at 0.3, then fewer false
alarms per hour at 0.3. Results should additionally report tIoU 0.7, boundary
errors and latency.

## Dataset caveats

- Fall bounds have approximately 0.5–1.0 second uncertainty.
- Falls are simulated on mats at three fixed positions.
- Streams are synthetic concatenations with 0.2 second crossfades.
- Pre-fall context is unusually separable from ADLs, creating a leakage-like
  shortcut that must be explicitly tested.
- Low-quality and `needs_review` events remain present; official-style local
  evaluation uses `--ignore-flagged`.

## Research safeguards

- Process every stream independently.
- Do not search for repeated content within/across streams or align against
  training segments.
- Do not use external copies of the original LIARA source recordings.
- Non-causal processing of one stream is allowed, but causal latency is still
  reported as a secondary deployment measure.
- Generate additional training streams only from permitted package segments,
  with every seed and split manifest recorded.

