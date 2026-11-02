# Nested decoder evaluation

To reduce validation-selection optimism, each validation stream is held out in
turn. Decoder hyperparameters are selected using the other four streams and are
then applied unchanged to the held-out stream. The Random Forest itself is fit
once using official training streams only.

| Feature variant | Nested ranking F1 | F1@0.1 | F1@0.3 | F1@0.5 | F1@0.7 | FA/hour @0.3 |
|---|---:|---:|---:|---:|---:|---:|
| All 82 features | 0.6575 | 0.8276 | 0.7034 | 0.4414 | 0.1931 | 13.18 |
| Without peak-motion features | **0.7080** | **0.8690** | **0.7862** | **0.4690** | 0.1931 | **10.25** |

Removing six peak-range-motion features improves both the nested score and the
false-alarm rate. The likely explanation is that argmax range bins are unstable
under multipath and low signal quality. This is evidence against keeping a
feature merely because it has an intuitive physical interpretation.

Held-out ranking F1 for the selected 76-feature model is 0.7273, 0.8034,
0.6667, 0.5238, and 0.6667 on `val_00` through `val_04`. `val_03` is the clear
failure-analysis priority. The variation also confirms that five streams are
too few for a narrow generalization claim.

This protocol makes the decoder assessment more defensible, but it is not a
fully nested model-development estimate: the decision to compare feature
variants was informed by validation results. A new untouched split or external
permitted benchmark remains necessary for a publication-grade final claim.
