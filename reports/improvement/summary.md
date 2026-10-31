# CR-SD results against the reproduced baselines

Test accuracy in %, mean ± std over seeds, validation selection. Paper = Table 6 (one run).

| Backbone | Setup | n | Sentiment | Topic | Paper |
|---|---|---:|---:|---:|---:|
| phobert | single | 15 | 82.27 ± 0.40 | 78.16 ± 0.58 | 81.75 / 78.65 |
| phobert | mtl | 15 | 82.39 ± 0.71 | 78.75 ± 0.36 | 82.88 / 79.25 |
| phobert | mtl_smart | 15 | 82.85 ± 0.24 | 79.05 ± 0.17 | 83.26 / 79.31 |
| phobert | mtl_mlm | 5 | 82.42 ± 0.26 | 78.38 ± 0.64 | 82.67 / 78.48 |
| phobert | mtl_mlm_smart | 5 | 82.77 ± 0.06 | 79.09 ± 0.18 | 83.66 / 79.49 |
| phobert | selfdistill | 2 | 83.30 ± 0.07 | 79.72 ± 0.02 | — |
| visobert | single | 15 | 82.30 ± 0.64 | 78.11 ± 0.42 | 82.78 / 77.94 |
| visobert | mtl | 15 | 82.04 ± 0.43 | 78.22 ± 0.55 | 82.63 / 78.56 |
| visobert | mtl_smart | 15 | 82.71 ± 0.18 | 79.35 ± 0.16 | 83.58 / 79.80 |
| visobert | mtl_mlm | 5 | 81.89 ± 0.29 | 78.04 ± 0.20 | 82.23 / 78.38 |
| visobert | mtl_mlm_smart | 5 | 82.83 ± 0.14 | 79.46 ± 0.23 | 83.12 / 79.25 |
| visobert | selfdistill | 2 | 83.03 ± 0.27 | 79.44 ± 0.16 | — |

## Improvement vs baseline (Δ of means, Welch p)

| Backbone | Setup | Task | n | EMA (reported) | Student | Δ vs mtl | p | Δ vs mtl_smart | p |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| phobert | selfdistill | sentiment | 2 | 83.30 ± 0.07 | 83.39 ± 0.05 | +0.91 | 0.000 | +0.45 | 0.003 |
| phobert | selfdistill | topic | 2 | 79.72 ± 0.02 | 79.49 ± 0.03 | +0.97 | 0.000 | +0.67 | 0.000 |
| visobert | selfdistill | sentiment | 2 | 83.03 ± 0.27 | 82.84 ± 0.27 | +0.98 | 0.056 | +0.31 | 0.333 |
| visobert | selfdistill | topic | 2 | 79.44 ± 0.16 | 79.34 ± 0.33 | +1.22 | 0.001 | +0.09 | 0.557 |
