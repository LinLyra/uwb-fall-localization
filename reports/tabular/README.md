# Physics-aware tabular baseline study

## Protocol

- Fit data: all 13 official training streams only.
- Evaluation data: 5 subject-disjoint official validation streams.
- Input: 2 s windows at 0.5 s hop.
- Features: 82 static-offset-resistant complex-CIR features covering temporal
  motion, coarse range bands, cross-radar energy, and peak-range displacement.
- Validation search: 144 combinations of probability threshold, merge gap, and
  symmetric boundary trim per model.
- Metric: official ranking mean of event F1 at tIoU 0.1, 0.3, and 0.5, with
  flagged events ignored exactly as in the official evaluator.

## Results

| Model | Ranking F1 | F1@0.1 | F1@0.3 | F1@0.5 | F1@0.7 | FA/hour @0.3 | Stream-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Official energy baseline | 0.3151 | 0.3582 | 0.3383 | 0.2488 | 0.0597 | 48.81 | not computed |
| Logistic v2 | 0.3699 | 0.5179 | 0.4036 | 0.1883 | 0.0538 | 54.18 | [0.2708, 0.4456] |
| HistGradientBoosting | 0.6667 | 0.8212 | 0.7285 | 0.4503 | 0.1192 | 14.15 | [0.5630, 0.7508] |
| Random Forest | **0.6912** | **0.8235** | **0.7500** | **0.5000** | **0.2059** | **8.79** | [0.5783, 0.7398] |

The Random Forest improves the validation ranking score by 0.3761 absolute over
the official energy baseline and reduces false alarms per hour by about 82%.
The remaining drop from tIoU 0.3 to 0.7 shows that boundary precision is still
the principal weakness.

## Interpretation limits

These are validation-tuned results, not held-out test estimates. Decoder
hyperparameters were selected on the same five validation streams used for the
table. The stream-level percentile bootstrap uses 2,000 replicates with seed
2027 and quantifies between-stream variability only; it does not remove model-
selection optimism. A publication claim must therefore add nested
leave-one-stream-out decoder selection or a new untouched evaluation split.

The large gain is credible enough to continue investigating but not sufficient
by itself to claim state of the art. Required next checks are feature-group
ablation, per-stream failure analysis, and nested selection.
