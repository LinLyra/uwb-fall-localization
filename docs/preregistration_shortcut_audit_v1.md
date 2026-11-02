# Preregistration v1: shortcut audit and robust UWB fall localization

Status: frozen before running the shortcut-control experiments described below  
Date: 2026-11-03 (Australia/Sydney)  
Scope: official ABC 2027 UWB training/validation package v1.1

## 1. Provenance and prior knowledge

This document is prospective for the shortcut controls, paired tests, boundary
analysis, and robustness experiments below. It is not prospective for the
existing baseline, tabular-model, feature-ablation, or decoder-selection
results: those results were already observed and are explicitly classified as
exploratory.

Known before freezing this protocol:

- official energy baseline ranking F1: 0.3151;
- validation-tuned Random Forest ranking F1: 0.6912;
- validation-tuned Random Forest without peak-motion features: 0.7083;
- leave-one-validation-stream-out decoder score for the no-peak variant: 0.7080;
- the no-peak feature choice was informed by validation ablation and therefore
  is not covered by the decoder-only nesting protocol;
- `val_03` has the weakest observed held-out score.

These values must not be presented as untouched confirmatory estimates.

## 2. Data and immutable evaluation population

- Fit split: 13 official training streams, 6.18 hours.
- Evaluation split: 5 subject-disjoint official validation streams, 2.05 hours.
- Validation events: 119 total; 52 ignored under the official
  `--ignore-flagged` rule; 67 evaluable.
- Hidden test streams and labels are not used for fitting, selection, or
  analysis.
- Primary metric: official ranking F1, the mean event F1 at tIoU 0.1, 0.3,
  and 0.5.
- Secondary metrics: F1 at each tIoU, false alarms/hour at tIoU 0.3, median
  start/end error, and F1 at tIoU 0.7.

No validation stream may be moved into model fitting for the confirmatory
audit. Combining train and validation is permitted only in a separately labeled
exploratory analysis after this protocol is complete.

## 3. Confirmatory questions and falsification criteria

### H1: useful performance depends on event motion, not static context alone

Control: preserve the pre-event/static context but destroy within-event temporal
motion by a fixed seeded permutation within each range bin and radar.

Falsification rule: H1 is rejected if the control retains at least 80% of the
unaltered model's paired stream-level ranking F1 improvement over the official
energy baseline.

### H2: pre-fall context alone is insufficient to localize future falls

Control: construct windows ending 0.5 seconds before each annotated event start
and matched negative windows from the same streams. No event or post-event
sample may enter the input.

Primary outcome: stream-grouped ROC-AUC for future-fall classification. The
event-localization F1 is not used because pre-fall-only inputs cannot identify
an end boundary without imposing one.

Falsification rule: H2 is rejected if the grouped out-of-fold AUC is at least
0.70 and its stream-bootstrap 95% interval excludes 0.50.

### H3: coarse range evidence is not merely a fixed-position lookup

Controls: circular-free translations of the residual range representation by
fixed offsets of -24, -16, -8, +8, +16, and +24 bins, with vacated bins zeroed;
and contiguous range masks covering 12.5%, 25%, and 50% of bins.

Outcome: paired change in ranking F1 and false alarms/hour versus zero shift.

Falsification rule: H3 is rejected if an 8-bin translation causes a median
paired ranking-F1 loss greater than 25% without a comparable loss under matched
random masking.

### H4: multi-radar evidence is robust to sensor loss and ordering

Controls: remove each radar separately; remove the highest-energy radar; and
permute radar order without retraining.

Outcome: paired stream-level ranking-F1 change. A model advertised as
radar-order invariant must change by no more than numerical tolerance under
permutation.

## 4. Statistical protocol

- Unit of resampling and permutation: complete validation stream.
- Report raw per-stream metrics and micro-aggregated official metrics.
- Model comparisons use paired stream-level bootstrap differences, not overlap
  between two marginal confidence intervals.
- Bootstrap replicates: 10,000; random seed: 2027; percentile 95% interval.
- Paired randomization test: all sign flips are enumerated when feasible for
  five streams; otherwise 10,000 seeded permutations.
- Family-wise correction: Holm correction across the four primary hypothesis
  families.
- Effect sizes and intervals are primary; p-values are secondary because five
  streams provide low statistical power.
- Tree-model stochasticity is evaluated at seeds 2027, 2028, 2029, 2030, and
  2031. Decoder settings are selected inside each held-out-stream fold.

The five-stream sample makes asymptotic claims inappropriate. Any conclusion
must be limited to this package's synthetic continuous-stream construction.

## 5. Boundary analysis

Before training a boundary neural network:

1. derive the relationship between event duration, start/end error, and tIoU;
2. plot the maximum symmetric error compatible with tIoU 0.3, 0.5, and 0.7;
3. report empirical event-duration quantiles;
4. simulate annotation perturbations of ±0.5 s and ±1.0 s, matching the
   documented timestamp uncertainty, and quantify the metric ceiling;
5. decide whether F1@0.7 is a meaningful optimization target under that noise.

## 6. Confirmatory and exploratory separation

Confirmatory outputs:

- four shortcut-control families above;
- paired stream-level differences and corrected tests;
- boundary-noise analysis;
- five-seed nested evaluation of the pre-specified 82-feature Random Forest.

Exploratory outputs:

- choosing the 76-feature no-peak variant;
- TCN, GRU, or state-space architecture search;
- new augmentation combinations;
- combining official train and validation streams;
- any analysis motivated by inspecting the new control results.

Exploratory findings may generate a v2 preregistration but may not be relabeled
as v1-confirmatory evidence.

## 7. Decision gates

- G1: do not train a larger temporal model until pre-fall and motion-destruction
  controls are complete.
- G2: if either control retains at least 80% of the observed gain, reframe the
  project around shortcut detection and mitigation rather than higher F1.
- G3: if controls fail to retain performance, proceed to boundary-aware models.
- G4: promote a method to the main result only when its full selection path is
  nested or evaluated on a genuinely untouched split.

## 8. Reproducibility contract

- Every confirmatory command, configuration, random seed, environment version,
  and output hash is retained.
- One command must regenerate each final table and figure from cached features.
- Raw data, caches, and checkpoints remain excluded from Git.
- Changes to this frozen document require a new version and Git tag; this file
  is never silently edited after the tag.
