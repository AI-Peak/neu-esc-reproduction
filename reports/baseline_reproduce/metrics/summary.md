# NEU-ESC reproduction summary

Test-set metrics in %, mean ± sample std over seeds. Δ = ours − paper. The paper reports one run per row.

| Backbone | Setup | Task | n | Acc | mF1 | wF1 | wF1 (author MTL eval) | Paper Acc / mF1 / wF1 | Δ Acc | Δ mF1 | Paper mF1 in seed range |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| phobert | mtl | sentiment | 15 | 82.39 ± 0.71 | 76.86 ± 0.63 | 82.39 ± 0.53 | 82.38 ± 0.92 | 82.88 / 75.99 / 82.85 | -0.49 | +0.87 | yes |
| phobert | mtl | topic | 15 | 78.75 ± 0.36 | 62.23 ± 0.41 | 78.38 ± 0.24 | 79.13 ± 0.52 | 79.25 / 62.11 / 79.59 | -0.50 | +0.12 | yes |
| phobert | mtl_mlm | sentiment | 5 | 82.42 ± 0.26 | 76.63 ± 0.49 | 82.38 ± 0.21 | 82.45 ± 0.51 | 82.67 / 77.03 / 82.72 | -0.25 | -0.40 | yes |
| phobert | mtl_mlm | topic | 5 | 78.38 ± 0.64 | 61.72 ± 0.85 | 78.11 ± 0.51 | 78.66 ± 0.77 | 78.48 / 62.33 / 78.44 | -0.10 | -0.61 | yes |
| phobert | mtl_mlm_smart | sentiment | 5 | 82.77 ± 0.06 | 76.20 ± 0.39 | 82.00 ± 0.16 | 83.55 ± 0.17 | 83.66 / 77.34 / 84.15 | -0.89 | -1.14 | no |
| phobert | mtl_mlm_smart | topic | 5 | 79.09 ± 0.18 | 62.21 ± 0.47 | 78.31 ± 0.18 | 79.88 ± 0.19 | 79.49 / 62.58 / 80.04 | -0.40 | -0.37 | yes |
| phobert | mtl_smart | sentiment | 15 | 82.85 ± 0.24 | 76.07 ± 0.57 | 82.02 ± 0.32 | 83.68 ± 0.22 | 83.26 / 77.52 / 83.66 | -0.41 | -1.45 | no |
| phobert | mtl_smart | topic | 15 | 79.05 ± 0.17 | 61.95 ± 0.60 | 78.23 ± 0.19 | 79.88 ± 0.22 | 79.31 / 62.79 / 80.04 | -0.26 | -0.84 | yes |
| phobert | single | sentiment | 15 | 82.27 ± 0.40 | 77.24 ± 0.37 | 82.27 ± 0.29 | — | 81.75 / 77.70 / 82.02 | +0.52 | -0.46 | yes |
| phobert | single | topic | 15 | 78.16 ± 0.58 | 61.49 ± 0.58 | 77.97 ± 0.44 | — | 78.65 / 62.73 / 78.36 | -0.49 | -1.24 | yes |
| visobert | mtl | sentiment | 15 | 82.04 ± 0.43 | 76.08 ± 0.76 | 81.93 ± 0.43 | 82.16 ± 0.48 | 82.63 / 76.17 / 82.58 | -0.59 | -0.09 | yes |
| visobert | mtl | topic | 15 | 78.22 ± 0.55 | 61.70 ± 0.60 | 77.95 ± 0.44 | 78.50 ± 0.69 | 78.56 / 61.66 / 78.68 | -0.34 | +0.04 | yes |
| visobert | mtl_mlm | sentiment | 5 | 81.89 ± 0.29 | 76.27 ± 0.29 | 81.86 ± 0.24 | 81.92 ± 0.38 | 82.23 / 75.15 / 82.70 | -0.34 | +1.12 | no |
| visobert | mtl_mlm | topic | 5 | 78.04 ± 0.20 | 61.58 ± 0.35 | 77.75 ± 0.20 | 78.33 ± 0.30 | 78.38 / 60.79 / 78.87 | -0.34 | +0.79 | no |
| visobert | mtl_mlm_smart | sentiment | 5 | 82.83 ± 0.14 | 76.09 ± 0.48 | 82.12 ± 0.27 | 83.54 ± 0.15 | 83.12 / 76.74 / 83.57 | -0.29 | -0.65 | no |
| visobert | mtl_mlm_smart | topic | 5 | 79.46 ± 0.23 | 62.33 ± 0.93 | 78.68 ± 0.38 | 80.25 ± 0.09 | 79.25 / 62.19 / 79.88 | +0.21 | +0.14 | yes |
| visobert | mtl_smart | sentiment | 15 | 82.71 ± 0.18 | 75.73 ± 0.43 | 81.88 ± 0.28 | 83.54 ± 0.15 | 83.58 / 77.17 / 84.19 | -0.87 | -1.44 | no |
| visobert | mtl_smart | topic | 15 | 79.35 ± 0.16 | 62.21 ± 0.41 | 78.49 ± 0.21 | 80.21 ± 0.15 | 79.80 / 63.13 / 80.55 | -0.45 | -0.92 | no |
| visobert | single | sentiment | 15 | 82.30 ± 0.64 | 76.51 ± 0.46 | 82.18 ± 0.44 | — | 82.78 / 75.79 / 82.17 | -0.48 | +0.72 | no |
| visobert | single | topic | 15 | 78.11 ± 0.42 | 61.32 ± 0.41 | 77.79 ± 0.29 | — | 77.94 / 60.65 / 78.06 | +0.17 | +0.67 | no |

## Gains between setups (macro-F1)

| Backbone | Task | Comparison | Δ mF1 (ours) | Welch p | Δ mF1 (paper) |
|---|---|---|---:|---:|---:|
| phobert | sentiment | single → mtl | -0.38 | 0.057 | -1.71 |
| phobert | sentiment | mtl → mtl_smart | -0.79 | 0.001 | +1.53 |
| phobert | sentiment | single → mtl_smart | -1.17 | 0.000 | -0.18 |
| phobert | sentiment | mtl → mtl_mlm | -0.24 | 0.409 | +1.04 |
| phobert | sentiment | mtl_mlm → mtl_mlm_smart | -0.43 | 0.161 | +0.31 |
| phobert | sentiment | single → mtl_mlm_smart | -1.04 | 0.001 | -0.36 |
| phobert | topic | single → mtl | +0.74 | 0.000 | -0.62 |
| phobert | topic | mtl → mtl_smart | -0.28 | 0.146 | +0.68 |
| phobert | topic | single → mtl_smart | +0.46 | 0.041 | +0.06 |
| phobert | topic | mtl → mtl_mlm | -0.51 | 0.254 | +0.22 |
| phobert | topic | mtl_mlm → mtl_mlm_smart | +0.49 | 0.298 | +0.25 |
| phobert | topic | single → mtl_mlm_smart | +0.72 | 0.021 | -0.15 |
| visobert | sentiment | single → mtl | -0.44 | 0.071 | +0.38 |
| visobert | sentiment | mtl → mtl_smart | -0.35 | 0.137 | +1.00 |
| visobert | sentiment | single → mtl_smart | -0.79 | 0.000 | +1.38 |
| visobert | sentiment | mtl → mtl_mlm | +0.20 | 0.410 | -1.02 |
| visobert | sentiment | mtl_mlm → mtl_mlm_smart | -0.18 | 0.495 | +1.59 |
| visobert | sentiment | single → mtl_mlm_smart | -0.42 | 0.135 | +0.95 |
| visobert | topic | single → mtl | +0.38 | 0.052 | +1.01 |
| visobert | topic | mtl → mtl_smart | +0.52 | 0.010 | +1.47 |
| visobert | topic | single → mtl_smart | +0.90 | 0.000 | +2.48 |
| visobert | topic | mtl → mtl_mlm | -0.12 | 0.591 | -0.87 |
| visobert | topic | mtl_mlm → mtl_mlm_smart | +0.75 | 0.150 | +1.40 |
| visobert | topic | single → mtl_mlm_smart | +1.01 | 0.070 | +1.54 |

## Runner version (SHA-256 of neu_esc_kaggle.py)

**WARNING: runs come from different runner versions; check per_run.csv before treating them as one cohort.**
- `2a2a25ae687a577c`: 40 runs
- `5a56bc356ba0f060`: 100 runs
