# Single-augmentation ablation

Each model adds augmented samples equal to 25% of the clean training-window
count. The clean data, model family, nested decoder protocol, and seed are held
fixed.

| Training variant | Ranking F1 | F1@0.3 | F1@0.7 | False alarms/hour |
|---|---:|---:|---:|---:|
| Clean invariant RF | 0.6619 | 0.7234 | 0.1560 | 11.23 |
| Range shift only | 0.6387 | 0.6993 | 0.1538 | 12.69 |
| Radar dropout only | 0.6462 | 0.6615 | 0.1846 | 9.76 |

Range-shift augmentation changes ranking F1 by -0.0232 with a paired interval
of [-0.0626, 0.0021]. It does not improve the clean result. Radar dropout changes
ranking F1 by -0.0158 with a wider interval of [-0.1031, 0.0649]. It lowers the
false-alarm rate and improves `val_01`–`val_03`, but substantially hurts
`val_00` and `val_04`.

Neither variant is promoted on clean performance. Radar dropout remains worth a
sensor-failure evaluation because its intended target is robustness rather than
clean F1. Range-shift augmentation needs smaller offsets or motion-centred
alignment; sampling large shifts from the first version is too aggressive.
