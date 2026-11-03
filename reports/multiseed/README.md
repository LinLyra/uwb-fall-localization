# Random-seed stability

The radar-order invariant Random Forest is evaluated with the same nested
decoder protocol at five seeds.

| Seed | Ranking F1 |
|---:|---:|
| 2027 | 0.6619 |
| 2028 | 0.6763 |
| 2029 | 0.6478 |
| 2030 | 0.6714 |
| 2031 | 0.6229 |

Mean ranking F1 is 0.6561; the observed range is 0.6229–0.6763. Model
stochasticity is smaller than the gap to the official energy baseline, but it
is large enough that a single-seed value should not be used as the headline
estimate. The 2027 result is close to the five-seed mean and is not an unusually
favourable seed.
