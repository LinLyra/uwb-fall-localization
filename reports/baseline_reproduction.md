# Official energy-baseline reproduction

Date: 2026-11-03 (Australia/Sydney)

## Environment

- Apple Silicon macOS
- Python 3.13.5
- Dependencies installed at the exact versions in the official
  `requirements.txt`
- Official training archive SHA-256:
  `795115faaca153b8c1ebb46c44530d79c9b28ca83cb616e0d87530744f4d743c`

## Command

From the extracted official training package:

```bash
PYTHONPATH=src ../../../.venv/bin/python -m abc2027.evaluate \
  --gt streams_events.csv \
  --pred sample_submission.csv \
  --streams streams/val \
  --ignore-flagged \
  --out ../../../reports/official_energy_baseline.json
```

## Result

| Metric | Reproduced | Published |
|---|---:|---:|
| Ranking F1 mean (0.1/0.3/0.5) | 0.3151 | 0.315 |
| F1 at tIoU 0.1 | 0.3582 | 0.358 |
| F1 at tIoU 0.3 | 0.3383 | 0.338 |
| F1 at tIoU 0.5 | 0.2488 | 0.249 |
| F1 at tIoU 0.7 | 0.0597 | 0.060 |

The official baseline is reproduced to the displayed precision. The complete
machine-readable report is stored in `official_energy_baseline.json`.

## Documentation scope trap

The command printed in the package README passes `--streams streams`. That
directory contains both training and validation streams, while
`sample_submission.csv` contains validation predictions only. With the shipped
evaluator this incorrectly counts unpredicted training falls as false negatives
and yields a ranking mean of 0.1637. Passing `--streams streams/val` correctly
scopes the evaluation and reproduces the published 0.3151 result.

All future experiments must explicitly select one split directory. This check
is treated as a protocol invariant rather than a cosmetic command-line detail.
