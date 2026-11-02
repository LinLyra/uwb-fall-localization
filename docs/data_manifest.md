# Official data manifest

This manifest records the immutable inputs used for reproducibility. The data
files themselves are excluded from Git and remain subject to their upstream
license.

## Acquisition

- Source: official ABC 2027 UWB-Fall Track public Google Drive folder
- Retrieved: 2026-11-03 (Australia/Sydney)
- Package documentation version: v1.1
- Archive integrity: both ZIP archives pass `unzip -t` with no errors

## SHA-256

| File | SHA-256 |
|---|---|
| `abc2027_uwb_train_v1.zip` | `795115faaca153b8c1ebb46c44530d79c9b28ca83cb616e0d87530744f4d743c` |
| `abc2027_uwb_test_v1.zip` | `ab70f9c6bcc05767c76246e3d481b22e44d8ee24a9cd2fdff360fbee266323e9` |
| `README_v1.1.md` | `fe37ff5aa794103d0dfe6bafa5725b47db24a468d9738f1fe90c308d0c233b3e` |
| `ABC2027_UWB_Challenge_Subtitled.mp4` | `ef9512c63d0b63d20535fc5a1d0fe2dda1fac1c67dcee69c5b8a79b91e0a83e5` |

## Safeguards

- Raw archives, extracted signals, model checkpoints, and run artifacts are
  ignored by Git.
- Validation remains subject-disjoint and is not merged into training during
  model selection.
- The hidden-label test package is used only for final inference and format
  checks, never for fitting or threshold selection.
- Dataset-specific licensing and the official stream-processing restrictions
  remain in force even though this is now an independent research project.
