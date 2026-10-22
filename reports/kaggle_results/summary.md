# NEU-ESC reproduction summary

Test-set metrics in %, mean ± sample std over seeds. Δ = ours − paper. The paper reports one run per row.

| Backbone | Setup | Task | n | Acc | mF1 | wF1 | wF1 (author MTL eval) | Paper Acc / mF1 / wF1 | Δ Acc | Δ mF1 | Paper mF1 in seed range |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| phobert | mtl | sentiment | 5 | 82.30 ± 0.77 | 76.62 ± 0.30 | 82.30 ± 0.56 | 82.31 ± 0.98 | 82.88 / 75.99 / 82.85 | -0.58 | +0.63 | no |
| phobert | mtl | topic | 5 | 78.80 ± 0.45 | 62.27 ± 0.42 | 78.36 ± 0.28 | 79.24 ± 0.68 | 79.25 / 62.11 / 79.59 | -0.45 | +0.16 | yes |
| phobert | mtl_smart | sentiment | 5 | 82.90 ± 0.33 | 76.29 ± 0.37 | 82.09 ± 0.33 | 83.71 ± 0.35 | 83.26 / 77.52 / 83.66 | -0.36 | -1.23 | no |
| phobert | mtl_smart | topic | 5 | 79.02 ± 0.20 | 62.32 ± 0.46 | 78.26 ± 0.25 | 79.77 ± 0.24 | 79.31 / 62.79 / 80.04 | -0.29 | -0.47 | no |
| phobert | single | sentiment | 5 | 82.02 ± 0.32 | 77.26 ± 0.41 | 82.11 ± 0.35 | — | 81.75 / 77.70 / 82.02 | +0.27 | -0.44 | yes |
| phobert | single | topic | 5 | 78.44 ± 0.74 | 61.63 ± 0.88 | 78.12 ± 0.57 | — | 78.65 / 62.73 / 78.36 | -0.21 | -1.10 | yes |
| visobert | mtl | sentiment | 5 | 82.30 ± 0.41 | 76.53 ± 0.71 | 82.15 ± 0.46 | 82.44 ± 0.40 | 82.63 / 76.17 / 82.58 | -0.33 | +0.36 | yes |
| visobert | mtl | topic | 5 | 77.77 ± 0.58 | 61.24 ± 0.30 | 77.61 ± 0.37 | 77.93 ± 0.80 | 78.56 / 61.66 / 78.68 | -0.79 | -0.42 | no |
| visobert | mtl_smart | sentiment | 5 | 82.77 ± 0.20 | 75.96 ± 0.45 | 81.99 ± 0.38 | 83.54 ± 0.13 | 83.58 / 77.17 / 84.19 | -0.81 | -1.21 | no |
| visobert | mtl_smart | topic | 5 | 79.28 ± 0.15 | 62.08 ± 0.56 | 78.40 ± 0.27 | 80.16 ± 0.11 | 79.80 / 63.13 / 80.55 | -0.52 | -1.05 | no |
| visobert | single | sentiment | 5 | 81.82 ± 0.64 | 76.44 ± 0.48 | 81.85 ± 0.38 | — | 82.78 / 75.79 / 82.17 | -0.96 | +0.65 | no |
| visobert | single | topic | 5 | 78.29 ± 0.52 | 61.61 ± 0.53 | 77.91 ± 0.41 | — | 77.94 / 60.65 / 78.06 | +0.35 | +0.96 | no |

## Gains between setups (macro-F1)

| Backbone | Task | Comparison | Δ mF1 (ours) | Welch p | Δ mF1 (paper) |
|---|---|---|---:|---:|---:|
| phobert | sentiment | single → mtl | -0.64 | 0.024 | -1.71 |
| phobert | sentiment | mtl → mtl_smart | -0.34 | 0.153 | +1.53 |
| phobert | sentiment | single → mtl_smart | -0.97 | 0.004 | -0.18 |
| phobert | topic | single → mtl | +0.64 | 0.194 | -0.62 |
| phobert | topic | mtl → mtl_smart | +0.04 | 0.881 | +0.68 |
| phobert | topic | single → mtl_smart | +0.68 | 0.174 | +0.06 |
| visobert | sentiment | single → mtl | +0.10 | 0.810 | +0.38 |
| visobert | sentiment | mtl → mtl_smart | -0.58 | 0.169 | +1.00 |
| visobert | sentiment | single → mtl_smart | -0.48 | 0.140 | +1.38 |
| visobert | topic | single → mtl | -0.37 | 0.216 | +1.01 |
| visobert | topic | mtl → mtl_smart | +0.84 | 0.026 | +1.47 |
| visobert | topic | single → mtl_smart | +0.47 | 0.213 | +2.48 |

## Runner version (SHA-256 of neu_esc_kaggle.py)

- `2a2a25ae687a577c`: 40 runs
