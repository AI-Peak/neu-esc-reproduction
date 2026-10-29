# Improvement cohort (neu_esc_caitien.py) vs reproduced baselines

Test-set metrics in %, mean ± sample std over seeds. Δ columns are macro-F1 / accuracy differences of means; p is Welch's t-test over seeds. sep_sum_ce is the baseline 2-task setup re-run by the improvement runner. Accuracy targets: sentiment > 83.7, topic > 79.8.

Factorial runner version(s): ac54c0d437ee

## Test macro-F1

| Backbone | Task | Setup | n | Acc | mF1 | Δ vs single | p | Δ vs sep_sum_ce | p | Acc target |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| phobert | sentiment | single | 5 | 82.02 ± 0.32 | 77.26 ± 0.41 | +0.00 | — | — | — | no |
| phobert | sentiment | mtl | 5 | 82.30 ± 0.77 | 76.62 ± 0.30 | -0.64 | 0.024 | — | — | no |
| phobert | sentiment | mtl_smart | 5 | 82.90 ± 0.33 | 76.29 ± 0.37 | -0.97 | 0.004 | — | — | no |
| phobert | sentiment | sep_sum_ce | 5 | 82.80 ± 0.58 | 76.52 ± 0.54 | -0.74 | 0.044 | +0.00 | — | no |
| phobert | sentiment | sep_sum_cbf | 5 | 81.79 ± 0.41 | 76.20 ± 0.52 | -1.06 | 0.008 | -0.33 | 0.359 | no |
| phobert | sentiment | sep_pcgrad_ce | 5 | 82.77 ± 0.14 | 76.83 ± 0.42 | -0.43 | 0.137 | +0.31 | 0.349 | no |
| phobert | sentiment | sep_pcgrad_cbf | 5 | 81.89 ± 0.47 | 75.99 ± 0.39 | -1.27 | 0.001 | -0.54 | 0.115 | no |
| phobert | sentiment | sep_patgrad_ce | 5 | 82.54 ± 0.50 | 76.64 ± 0.60 | -0.62 | 0.095 | +0.11 | 0.766 | no |
| phobert | sentiment | sep_patgrad_cbf | 5 | 82.08 ± 0.35 | 76.36 ± 0.21 | -0.90 | 0.005 | -0.16 | 0.561 | no |
| phobert | sentiment | fac_sum_ce | 5 | 82.59 ± 0.33 | 76.67 ± 0.37 | -0.58 | 0.045 | +0.15 | 0.624 | no |
| phobert | sentiment | fac_sum_cbf | 5 | 82.28 ± 0.40 | 76.40 ± 0.61 | -0.86 | 0.034 | -0.12 | 0.748 | no |
| phobert | sentiment | fac_pcgrad_ce | 5 | 82.59 ± 0.35 | 76.61 ± 0.39 | -0.65 | 0.033 | +0.09 | 0.781 | no |
| phobert | sentiment | fac_pcgrad_cbf | 5 | 82.49 ± 0.27 | 76.60 ± 0.45 | -0.66 | 0.041 | +0.08 | 0.813 | no |
| phobert | sentiment | fac_patgrad_ce | 5 | 82.54 ± 0.66 | 76.60 ± 0.29 | -0.66 | 0.020 | +0.07 | 0.803 | no |
| phobert | sentiment | fac_patgrad_cbf | 5 | 82.52 ± 0.43 | 76.64 ± 0.46 | -0.62 | 0.056 | +0.12 | 0.716 | no |
| phobert | topic | single | 5 | 78.44 ± 0.74 | 61.63 ± 0.88 | +0.00 | — | — | — | no |
| phobert | topic | mtl | 5 | 78.80 ± 0.45 | 62.27 ± 0.42 | +0.64 | 0.194 | — | — | no |
| phobert | topic | mtl_smart | 5 | 79.02 ± 0.20 | 62.32 ± 0.46 | +0.68 | 0.174 | — | — | no |
| phobert | topic | sep_sum_ce | 5 | 78.65 ± 0.28 | 62.09 ± 0.68 | +0.46 | 0.384 | +0.00 | — | no |
| phobert | topic | sep_sum_cbf | 5 | 77.90 ± 0.28 | 62.27 ± 0.62 | +0.64 | 0.225 | +0.18 | 0.671 | no |
| phobert | topic | sep_pcgrad_ce | 5 | 78.72 ± 0.49 | 61.64 ± 0.46 | +0.00 | 0.995 | -0.46 | 0.254 | no |
| phobert | topic | sep_pcgrad_cbf | 5 | 77.84 ± 0.56 | 62.23 ± 0.72 | +0.59 | 0.277 | +0.14 | 0.766 | no |
| phobert | topic | sep_patgrad_ce | 5 | 78.21 ± 0.36 | 61.95 ± 0.30 | +0.32 | 0.476 | -0.14 | 0.692 | no |
| phobert | topic | sep_patgrad_cbf | 5 | 77.63 ± 0.30 | 61.96 ± 0.69 | +0.32 | 0.536 | -0.14 | 0.762 | no |
| phobert | topic | fac_sum_ce | 5 | 78.57 ± 0.55 | 62.08 ± 0.76 | +0.45 | 0.412 | -0.01 | 0.988 | no |
| phobert | topic | fac_sum_cbf | 5 | 78.41 ± 0.66 | 61.96 ± 0.36 | +0.33 | 0.470 | -0.13 | 0.722 | no |
| phobert | topic | fac_pcgrad_ce | 5 | 78.19 ± 0.64 | 62.08 ± 0.55 | +0.45 | 0.365 | -0.01 | 0.985 | no |
| phobert | topic | fac_pcgrad_cbf | 5 | 78.39 ± 0.64 | 61.79 ± 0.43 | +0.15 | 0.738 | -0.30 | 0.425 | no |
| phobert | topic | fac_patgrad_ce | 5 | 78.47 ± 0.72 | 61.75 ± 0.62 | +0.11 | 0.818 | -0.34 | 0.426 | no |
| phobert | topic | fac_patgrad_cbf | 5 | 78.21 ± 0.53 | 61.94 ± 0.17 | +0.31 | 0.482 | -0.15 | 0.652 | no |
| visobert | sentiment | single | 5 | 81.82 ± 0.64 | 76.44 ± 0.48 | +0.00 | — | — | — | no |
| visobert | sentiment | mtl | 5 | 82.30 ± 0.41 | 76.53 ± 0.71 | +0.10 | 0.810 | — | — | no |
| visobert | sentiment | mtl_smart | 5 | 82.77 ± 0.20 | 75.96 ± 0.45 | -0.48 | 0.140 | — | — | no |
| visobert | sentiment | sep_sum_ce | 5 | 82.29 ± 0.43 | 76.52 ± 0.71 | +0.08 | 0.833 | +0.00 | — | no |
| visobert | sentiment | sep_sum_cbf | 5 | 82.03 ± 0.32 | 76.06 ± 0.49 | -0.38 | 0.255 | -0.46 | 0.272 | no |
| visobert | sentiment | sep_pcgrad_ce | 5 | 82.27 ± 0.37 | 76.53 ± 0.63 | +0.09 | 0.797 | +0.01 | 0.980 | no |
| visobert | sentiment | sep_pcgrad_cbf | 5 | 82.11 ± 0.34 | 76.05 ± 0.38 | -0.38 | 0.202 | -0.47 | 0.242 | no |
| visobert | sentiment | sep_patgrad_ce | 5 | 82.27 ± 0.43 | 76.50 ± 0.73 | +0.06 | 0.873 | -0.02 | 0.968 | no |
| visobert | sentiment | sep_patgrad_cbf | 5 | 82.08 ± 0.30 | 76.03 ± 0.53 | -0.41 | 0.237 | -0.49 | 0.252 | no |
| visobert | sentiment | fac_sum_ce | 5 | 82.05 ± 0.21 | 76.30 ± 0.50 | -0.14 | 0.662 | -0.22 | 0.581 | no |
| visobert | sentiment | fac_sum_cbf | 5 | 81.96 ± 0.52 | 76.15 ± 0.65 | -0.29 | 0.448 | -0.37 | 0.412 | no |
| visobert | sentiment | fac_pcgrad_ce | 5 | 81.99 ± 0.21 | 76.15 ± 0.68 | -0.28 | 0.472 | -0.37 | 0.429 | no |
| visobert | sentiment | fac_pcgrad_cbf | 5 | 81.97 ± 0.43 | 76.11 ± 0.51 | -0.33 | 0.329 | -0.41 | 0.329 | no |
| visobert | sentiment | fac_patgrad_ce | 5 | 81.98 ± 0.26 | 76.26 ± 0.64 | -0.17 | 0.645 | -0.26 | 0.567 | no |
| visobert | sentiment | fac_patgrad_cbf | 5 | 82.07 ± 0.48 | 76.07 ± 0.44 | -0.37 | 0.243 | -0.45 | 0.268 | no |
| visobert | topic | single | 5 | 78.29 ± 0.52 | 61.61 ± 0.53 | +0.00 | — | — | — | no |
| visobert | topic | mtl | 5 | 77.77 ± 0.58 | 61.24 ± 0.30 | -0.37 | 0.216 | — | — | no |
| visobert | topic | mtl_smart | 5 | 79.28 ± 0.15 | 62.08 ± 0.56 | +0.47 | 0.213 | — | — | no |
| visobert | topic | sep_sum_ce | 5 | 77.79 ± 0.58 | 61.27 ± 0.30 | -0.35 | 0.248 | +0.00 | — | no |
| visobert | topic | sep_sum_cbf | 5 | 77.95 ± 0.30 | 61.87 ± 0.40 | +0.26 | 0.408 | +0.61 | 0.029 | no |
| visobert | topic | sep_pcgrad_ce | 5 | 77.78 ± 0.54 | 61.24 ± 0.24 | -0.37 | 0.203 | -0.03 | 0.879 | no |
| visobert | topic | sep_pcgrad_cbf | 5 | 77.80 ± 0.44 | 61.78 ± 0.30 | +0.16 | 0.568 | +0.51 | 0.029 | no |
| visobert | topic | sep_patgrad_ce | 5 | 77.80 ± 0.57 | 61.29 ± 0.32 | -0.32 | 0.286 | +0.03 | 0.899 | no |
| visobert | topic | sep_patgrad_cbf | 5 | 77.99 ± 0.32 | 61.85 ± 0.37 | +0.24 | 0.433 | +0.58 | 0.026 | no |
| visobert | topic | fac_sum_ce | 5 | 77.77 ± 0.70 | 61.16 ± 0.51 | -0.46 | 0.200 | -0.11 | 0.686 | no |
| visobert | topic | fac_sum_cbf | 5 | 77.93 ± 0.42 | 61.09 ± 0.29 | -0.52 | 0.099 | -0.18 | 0.375 | no |
| visobert | topic | fac_pcgrad_ce | 5 | 77.72 ± 0.66 | 61.09 ± 0.60 | -0.52 | 0.185 | -0.17 | 0.586 | no |
| visobert | topic | fac_pcgrad_cbf | 5 | 77.89 ± 0.26 | 61.45 ± 0.15 | -0.16 | 0.539 | +0.18 | 0.272 | no |
| visobert | topic | fac_patgrad_ce | 5 | 77.77 ± 0.69 | 61.21 ± 0.61 | -0.41 | 0.293 | -0.06 | 0.849 | no |
| visobert | topic | fac_patgrad_cbf | 5 | 77.84 ± 0.17 | 61.31 ± 0.20 | -0.31 | 0.279 | +0.04 | 0.811 | no |

## Test accuracy: deltas

| Backbone | Task | Setup | Δ acc vs single | p | Δ acc vs sep_sum_ce | p |
|---|---|---|---:|---:|---:|---:|
| phobert | sentiment | single | +0.00 | — | — | — |
| phobert | sentiment | mtl | +0.28 | 0.481 | — | — |
| phobert | sentiment | mtl_smart | +0.88 | 0.003 | — | — |
| phobert | sentiment | sep_sum_ce | +0.77 | 0.039 | +0.00 | — |
| phobert | sentiment | sep_sum_cbf | -0.23 | 0.354 | -1.00 | 0.016 |
| phobert | sentiment | sep_pcgrad_ce | +0.75 | 0.004 | -0.02 | 0.932 |
| phobert | sentiment | sep_pcgrad_cbf | -0.14 | 0.610 | -0.91 | 0.028 |
| phobert | sentiment | sep_patgrad_ce | +0.52 | 0.093 | -0.26 | 0.476 |
| phobert | sentiment | sep_patgrad_cbf | +0.05 | 0.804 | -0.72 | 0.052 |
| phobert | sentiment | fac_sum_ce | +0.56 | 0.026 | -0.21 | 0.506 |
| phobert | sentiment | fac_sum_cbf | +0.26 | 0.296 | -0.52 | 0.145 |
| phobert | sentiment | fac_pcgrad_ce | +0.57 | 0.028 | -0.20 | 0.528 |
| phobert | sentiment | fac_pcgrad_cbf | +0.47 | 0.039 | -0.31 | 0.332 |
| phobert | sentiment | fac_patgrad_ce | +0.51 | 0.170 | -0.26 | 0.528 |
| phobert | sentiment | fac_patgrad_cbf | +0.49 | 0.075 | -0.28 | 0.411 |
| phobert | topic | single | +0.00 | — | — | — |
| phobert | topic | mtl | +0.36 | 0.388 | — | — |
| phobert | topic | mtl_smart | +0.58 | 0.159 | — | — |
| phobert | topic | sep_sum_ce | +0.21 | 0.577 | +0.00 | — |
| phobert | topic | sep_sum_cbf | -0.54 | 0.191 | -0.75 | 0.003 |
| phobert | topic | sep_pcgrad_ce | +0.28 | 0.502 | +0.07 | 0.791 |
| phobert | topic | sep_pcgrad_cbf | -0.60 | 0.190 | -0.81 | 0.029 |
| phobert | topic | sep_patgrad_ce | -0.23 | 0.557 | -0.44 | 0.064 |
| phobert | topic | sep_patgrad_cbf | -0.81 | 0.071 | -1.02 | 0.001 |
| phobert | topic | fac_sum_ce | +0.13 | 0.757 | -0.08 | 0.786 |
| phobert | topic | fac_sum_cbf | -0.03 | 0.942 | -0.24 | 0.476 |
| phobert | topic | fac_pcgrad_ce | -0.25 | 0.578 | -0.47 | 0.191 |
| phobert | topic | fac_pcgrad_cbf | -0.05 | 0.915 | -0.26 | 0.439 |
| phobert | topic | fac_patgrad_ce | +0.03 | 0.954 | -0.18 | 0.616 |
| phobert | topic | fac_patgrad_cbf | -0.23 | 0.596 | -0.44 | 0.156 |
| visobert | sentiment | single | +0.00 | — | — | — |
| visobert | sentiment | mtl | +0.47 | 0.209 | — | — |
| visobert | sentiment | mtl_smart | +0.94 | 0.027 | — | — |
| visobert | sentiment | sep_sum_ce | +0.47 | 0.216 | +0.00 | — |
| visobert | sentiment | sep_sum_cbf | +0.21 | 0.543 | -0.26 | 0.307 |
| visobert | sentiment | sep_pcgrad_ce | +0.44 | 0.226 | -0.03 | 0.917 |
| visobert | sentiment | sep_pcgrad_cbf | +0.28 | 0.413 | -0.18 | 0.477 |
| visobert | sentiment | sep_patgrad_ce | +0.45 | 0.231 | -0.02 | 0.948 |
| visobert | sentiment | sep_patgrad_cbf | +0.26 | 0.447 | -0.21 | 0.395 |
| visobert | sentiment | fac_sum_ce | +0.23 | 0.485 | -0.24 | 0.304 |
| visobert | sentiment | fac_sum_cbf | +0.13 | 0.728 | -0.34 | 0.302 |
| visobert | sentiment | fac_pcgrad_ce | +0.17 | 0.605 | -0.30 | 0.211 |
| visobert | sentiment | fac_pcgrad_cbf | +0.15 | 0.674 | -0.32 | 0.277 |
| visobert | sentiment | fac_patgrad_ce | +0.16 | 0.624 | -0.31 | 0.216 |
| visobert | sentiment | fac_patgrad_cbf | +0.25 | 0.505 | -0.22 | 0.474 |
| visobert | topic | single | +0.00 | — | — | — |
| visobert | topic | mtl | -0.51 | 0.179 | — | — |
| visobert | topic | mtl_smart | +1.00 | 0.011 | — | — |
| visobert | topic | sep_sum_ce | -0.49 | 0.196 | +0.00 | — |
| visobert | topic | sep_sum_cbf | -0.34 | 0.253 | +0.15 | 0.617 |
| visobert | topic | sep_pcgrad_ce | -0.51 | 0.170 | -0.01 | 0.974 |
| visobert | topic | sep_pcgrad_cbf | -0.49 | 0.150 | +0.01 | 0.986 |
| visobert | topic | sep_patgrad_ce | -0.49 | 0.196 | +0.00 | 0.994 |
| visobert | topic | sep_patgrad_cbf | -0.30 | 0.318 | +0.20 | 0.531 |
| visobert | topic | fac_sum_ce | -0.51 | 0.226 | -0.02 | 0.960 |
| visobert | topic | fac_sum_cbf | -0.36 | 0.265 | +0.13 | 0.689 |
| visobert | topic | fac_pcgrad_ce | -0.57 | 0.173 | -0.08 | 0.853 |
| visobert | topic | fac_pcgrad_cbf | -0.40 | 0.181 | +0.10 | 0.746 |
| visobert | topic | fac_patgrad_ce | -0.51 | 0.225 | -0.02 | 0.960 |
| visobert | topic | fac_patgrad_cbf | -0.44 | 0.135 | +0.05 | 0.857 |

## Main effects (paired by seed and the other two factors)

Each difference compares two setups that differ only in this factor, for the same seed; p is a one-sample t-test of those differences against 0.

| Backbone | Task | Contrast | pairs | Δ mF1 | p | Δ acc | p |
|---|---|---|---:|---:|---:|---:|---:|
| phobert | sentiment | head: sep → fac | 30 | 0.17 ± 0.61 | 0.149 | 0.19 ± 0.59 | 0.087 |
| phobert | sentiment | opt: sum → pcgrad | 20 | 0.06 ± 0.57 | 0.657 | 0.07 ± 0.45 | 0.481 |
| phobert | sentiment | opt: sum → patgrad | 20 | 0.11 ± 0.64 | 0.452 | 0.05 ± 0.48 | 0.619 |
| phobert | sentiment | loss: ce → cbf | 30 | -0.28 ± 0.64 | 0.024 | -0.46 ± 0.68 | 0.001 |
| phobert | topic | head: sep → fac | 30 | -0.09 ± 0.69 | 0.490 | 0.21 ± 0.74 | 0.127 |
| phobert | topic | opt: sum → pcgrad | 20 | -0.17 ± 0.66 | 0.265 | -0.10 ± 0.40 | 0.277 |
| phobert | topic | opt: sum → patgrad | 20 | -0.20 ± 0.57 | 0.129 | -0.25 ± 0.33 | 0.003 |
| phobert | topic | loss: ce → cbf | 30 | 0.09 ± 0.81 | 0.538 | -0.40 ± 0.64 | 0.002 |
| visobert | sentiment | head: sep → fac | 30 | -0.11 ± 0.62 | 0.340 | -0.17 ± 0.40 | 0.027 |
| visobert | sentiment | opt: sum → pcgrad | 20 | -0.04 ± 0.42 | 0.644 | 0.00 ± 0.26 | 0.969 |
| visobert | sentiment | opt: sum → patgrad | 20 | -0.04 ± 0.39 | 0.642 | 0.02 ± 0.21 | 0.652 |
| visobert | sentiment | loss: ce → cbf | 30 | -0.30 ± 0.65 | 0.018 | -0.11 ± 0.45 | 0.211 |
| visobert | topic | head: sep → fac | 30 | -0.33 ± 0.42 | 0.000 | -0.03 ± 0.37 | 0.652 |
| visobert | topic | opt: sum → pcgrad | 20 | 0.04 ± 0.33 | 0.560 | -0.06 ± 0.22 | 0.209 |
| visobert | topic | opt: sum → patgrad | 20 | 0.07 ± 0.26 | 0.250 | -0.01 ± 0.16 | 0.799 |
| visobert | topic | loss: ce → cbf | 30 | 0.35 ± 0.63 | 0.005 | 0.13 ± 0.55 | 0.216 |

## Factorized head: marginal vs joint-argmax inference (test)

| Backbone | Task | Setup | mF1 marginal | mF1 joint | Acc marginal | Acc joint | val prefers joint |
|---|---|---|---:|---:|---:|---:|---:|
| phobert | sentiment | fac_sum_ce | 76.67 ± 0.37 | 76.60 ± 0.23 | 82.59 ± 0.33 | 82.55 ± 0.33 | 2/5 |
| phobert | sentiment | fac_sum_cbf | 76.40 ± 0.61 | 76.32 ± 0.69 | 82.28 ± 0.40 | 82.27 ± 0.33 | 1/5 |
| phobert | sentiment | fac_pcgrad_ce | 76.61 ± 0.39 | 76.54 ± 0.37 | 82.59 ± 0.35 | 82.55 ± 0.42 | 3/5 |
| phobert | sentiment | fac_pcgrad_cbf | 76.60 ± 0.45 | 76.52 ± 0.39 | 82.49 ± 0.27 | 82.50 ± 0.31 | 2/5 |
| phobert | sentiment | fac_patgrad_ce | 76.60 ± 0.29 | 76.56 ± 0.28 | 82.54 ± 0.66 | 82.51 ± 0.64 | 2/5 |
| phobert | sentiment | fac_patgrad_cbf | 76.64 ± 0.46 | 76.52 ± 0.38 | 82.52 ± 0.43 | 82.46 ± 0.40 | 3/5 |
| phobert | topic | fac_sum_ce | 62.08 ± 0.76 | 62.08 ± 0.78 | 78.57 ± 0.55 | 78.52 ± 0.50 | 3/5 |
| phobert | topic | fac_sum_cbf | 61.96 ± 0.36 | 61.96 ± 0.31 | 78.41 ± 0.66 | 78.39 ± 0.66 | 1/5 |
| phobert | topic | fac_pcgrad_ce | 62.08 ± 0.55 | 62.05 ± 0.64 | 78.19 ± 0.64 | 78.10 ± 0.57 | 1/5 |
| phobert | topic | fac_pcgrad_cbf | 61.79 ± 0.43 | 61.85 ± 0.48 | 78.39 ± 0.64 | 78.36 ± 0.64 | 4/5 |
| phobert | topic | fac_patgrad_ce | 61.75 ± 0.62 | 61.90 ± 0.64 | 78.47 ± 0.72 | 78.49 ± 0.72 | 3/5 |
| phobert | topic | fac_patgrad_cbf | 61.94 ± 0.17 | 61.81 ± 0.30 | 78.21 ± 0.53 | 78.15 ± 0.59 | 0/5 |
| visobert | sentiment | fac_sum_ce | 76.30 ± 0.50 | 76.28 ± 0.48 | 82.05 ± 0.21 | 82.02 ± 0.19 | 3/5 |
| visobert | sentiment | fac_sum_cbf | 76.15 ± 0.65 | 76.17 ± 0.74 | 81.96 ± 0.52 | 81.96 ± 0.49 | 2/5 |
| visobert | sentiment | fac_pcgrad_ce | 76.15 ± 0.68 | 76.13 ± 0.67 | 81.99 ± 0.21 | 81.97 ± 0.16 | 1/5 |
| visobert | sentiment | fac_pcgrad_cbf | 76.11 ± 0.51 | 76.08 ± 0.49 | 81.97 ± 0.43 | 81.96 ± 0.42 | 2/5 |
| visobert | sentiment | fac_patgrad_ce | 76.26 ± 0.64 | 76.25 ± 0.57 | 81.98 ± 0.26 | 81.97 ± 0.21 | 2/5 |
| visobert | sentiment | fac_patgrad_cbf | 76.07 ± 0.44 | 76.02 ± 0.39 | 82.07 ± 0.48 | 82.05 ± 0.45 | 2/5 |
| visobert | topic | fac_sum_ce | 61.16 ± 0.51 | 61.18 ± 0.36 | 77.77 ± 0.70 | 77.75 ± 0.67 | 3/5 |
| visobert | topic | fac_sum_cbf | 61.09 ± 0.29 | 61.08 ± 0.27 | 77.93 ± 0.42 | 77.93 ± 0.44 | 1/5 |
| visobert | topic | fac_pcgrad_ce | 61.09 ± 0.60 | 61.18 ± 0.48 | 77.72 ± 0.66 | 77.75 ± 0.62 | 3/5 |
| visobert | topic | fac_pcgrad_cbf | 61.45 ± 0.15 | 61.34 ± 0.23 | 77.89 ± 0.26 | 77.88 ± 0.26 | 3/5 |
| visobert | topic | fac_patgrad_ce | 61.21 ± 0.61 | 61.24 ± 0.46 | 77.77 ± 0.69 | 77.75 ± 0.67 | 5/5 |
| visobert | topic | fac_patgrad_cbf | 61.31 ± 0.20 | 61.20 ± 0.16 | 77.84 ± 0.17 | 77.80 ± 0.16 | 1/5 |

## Gradient conflict diagnostics (surgery runs, mean over epochs and seeds)

global = whole encoder; per layer = share of (step, layer group) pairs with cos(g_s, g_t) < 0.

| Backbone | Setup | global cos | global conflict rate | per-layer conflict rate | projections / step | PaTGrad protects sentiment | layers chosen (seed 42) |
|---|---|---:|---:|---:|---:|---:|---|
| phobert | sep_pcgrad_ce | 0.014 | 0.42 | 0.44 | 5.70 | — | — |
| phobert | sep_pcgrad_cbf | 0.011 | 0.44 | 0.44 | 5.69 | — | — |
| phobert | sep_patgrad_ce | 0.015 | 0.42 | 0.44 | 2.52 | 0.95 | layer06, layer07, layer08, layer09, layer10, layer11 |
| phobert | sep_patgrad_cbf | 0.012 | 0.43 | 0.44 | 2.52 | 1.00 | layer06, layer07, layer08, layer09, layer10, layer11 |
| phobert | fac_pcgrad_ce | 0.028 | 0.36 | 0.36 | 4.71 | — | — |
| phobert | fac_pcgrad_cbf | 0.027 | 0.38 | 0.38 | 4.95 | — | — |
| phobert | fac_patgrad_ce | 0.028 | 0.36 | 0.36 | 2.06 | 0.96 | emb, layer00, layer01, layer09, layer10, layer11 |
| phobert | fac_patgrad_cbf | 0.027 | 0.38 | 0.38 | 2.14 | 0.96 | emb, layer00, layer01, layer02, layer03, layer08 |
| visobert | sep_pcgrad_ce | 0.000 | 0.52 | 0.55 | 7.13 | — | — |
| visobert | sep_pcgrad_cbf | 0.004 | 0.47 | 0.48 | 6.25 | — | — |
| visobert | sep_patgrad_ce | 0.000 | 0.52 | 0.55 | 3.29 | 0.95 | layer06, layer07, layer08, layer09, layer10, layer11 |
| visobert | sep_patgrad_cbf | 0.004 | 0.47 | 0.48 | 2.77 | 0.99 | layer06, layer07, layer08, layer09, layer10, layer11 |
| visobert | fac_pcgrad_ce | 0.019 | 0.37 | 0.36 | 4.70 | — | — |
| visobert | fac_pcgrad_cbf | 0.014 | 0.43 | 0.43 | 5.54 | — | — |
| visobert | fac_patgrad_ce | 0.019 | 0.37 | 0.36 | 2.04 | 0.93 | layer03, layer06, layer07, layer08, layer10, layer11 |
| visobert | fac_patgrad_cbf | 0.014 | 0.43 | 0.43 | 2.45 | 0.94 | layer06, layer07, layer08, layer09, layer10, layer11 |

## GPU hours per run (T4)

| Setup | mean h |
|---|---:|
| sep_sum_ce | 0.62 |
| sep_sum_cbf | 0.65 |
| sep_pcgrad_ce | 1.02 |
| sep_pcgrad_cbf | 1.05 |
| sep_patgrad_ce | 1.02 |
| sep_patgrad_cbf | 1.05 |
| fac_sum_ce | 0.62 |
| fac_sum_cbf | 0.65 |
| fac_pcgrad_ce | 1.39 |
| fac_pcgrad_cbf | 1.39 |
| fac_patgrad_ce | 1.38 |
| fac_patgrad_cbf | 1.41 |
