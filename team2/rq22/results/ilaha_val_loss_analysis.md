# RQ 2.2 — GW-1B Validation Loss Analysis

## 1. Objective

Estimate the validation loss of GW-1B (1.15B parameters,
20B training tokens) using the completed RQ 2.2 proxy
training runs.

## 2. Dataset

All 12 primary training runs are complete.

| Model | Actual parameters | t10 loss | t20 loss | t30 loss |
|---|---:|---:|---:|---:|
| 50M | 50,213,376 | 3.277724 | 3.151757 | 3.094283 |
| 100M | 100,092,672 | 3.034312 | 2.932259 | 2.882846 |
| 200M | 200,573,952 | 2.803991 | 2.699287 | 2.653566 |
| 350M | 353,614,080 | 2.667833 | 2.593308 | 2.553295 |

Actual parameter counts, actual training token counts,
and full-precision validation losses were extracted
from the training summary.json files.

Additional seeds and the 50M-576x9 architecture
variant were excluded from the primary fit.

## 3. Methodology

The scaling law is:

L(N, D) = E + A / N^alpha + B / D^beta

where N is measured in millions of parameters and D
in billions of training tokens.

Two fitting methods were evaluated:

- Nonlinear Least Squares on raw validation loss.
- Huber fitting on raw validation loss, with f_scale=0.01.

Both methods use bounded optimization with 36 initial
parameter configurations and max_nfev=20000.
The successful fit with the lowest objective value
is selected for each method.

The target configuration is N=1150 and D=20.

## 4. Final Fitting Results

| Metric | Least Squares | Huber |
|---|---:|---:|
| GW-1B predicted val/loss | 2.368566 | 2.360957 |
| Training RMSE | 0.012032 | 0.012977 |
| 350M holdout mean error | 1.819% | 1.844% |

### Fitted Parameters

| Parameter | Least Squares | Huber |
|---|---:|---:|
| E | 1.717101 | 1.602749 |
| A | 2.342587 | 2.177302 |
| alpha | 0.310102 | 0.223655 |
| B | 0.746100 | 0.647485 |
| beta | 0.218180 | 0.247977 |

Both methods produce similar GW-1B predictions,
despite differences in their fitted coefficients.

## 5. Comparison with Preliminary Results

The preliminary analysis used 10 completed runs.
The final analysis includes the previously unavailable
200M-t30 and 350M-t30 runs.

| Method | Preliminary (10 runs) | Final (12 runs) |
|---|---:|---:|
| Least Squares | 2.358094 | 2.368566 |
| Huber | 2.358014 | 2.360957 |

The preliminary predictions were reproduced using
the reconstructed fitting script:

- Least Squares: 2.3580941913620252
- Huber: 2.358013658248254

Both match the previously reported predictions to
six decimal places.

The addition of the final two runs changes the
Least Squares prediction by approximately 0.44%
and the Huber prediction by approximately 0.12%.

## 6. 350M Holdout Validation

The scaling law was fitted using the nine 50M,
100M, and 200M runs. The three 350M runs were
excluded from fitting and used for evaluation.

| Run | Actual loss | LS prediction | Huber prediction | LS error | Huber error |
|---|---:|---:|---:|---:|---:|
| 350M-t10 | 2.667833 | 2.632755 | 2.632887 | 1.315% | 1.310% |
| 350M-t20 | 2.593308 | 2.544088 | 2.543301 | 1.898% | 1.928% |
| 350M-t30 | 2.553295 | 2.496006 | 2.494698 | 2.244% | 2.295% |
| Mean error | - | - | - | 1.819% | 1.844% |

Both methods achieve a mean absolute percentage
error below 2%. However, both systematically
underestimate the observed losses for the three
held-out 350M runs.

## 7. Leave-One-Model-Size-Out Sensitivity

Each model size was excluded in turn, and the
scaling law was refitted using the remaining nine runs.

| Excluded model size | Runs used | Least Squares | Huber |
|---|---:|---:|---:|
| None | 12 | 2.368566 | 2.360957 |
| 50M | 9 | 2.455826 | 2.455826 |
| 100M | 9 | 2.436847 | 2.436847 |
| 200M | 9 | 2.342985 | 2.342985 |
| 350M | 9 | 2.246204 | 2.245481 |

The overall sensitivity range is approximately
2.245–2.456. This is a sensitivity range, not a
statistical confidence interval.

Excluding the 350M runs produces a substantially
lower GW-1B prediction, highlighting the importance
of larger proxy models for extrapolation.

## 8. Interpretation and Limitations

- Validation loss decreases consistently with
  increasing model size and training token budget.
- The two fitting methods yield similar full-data
  predictions.
- The final predictions remain close to the
  preliminary estimates.
- The 350M holdout results indicate reasonable
  extrapolation performance within the proxy ladder,
  but also reveal systematic loss underestimation.
- The leave-one-size-out results demonstrate
  sensitivity to the available model sizes.
- The primary analysis contains only 12 runs and
  four distinct model sizes.
- Additional seeds and architecture variants are
  not included in the primary fit.
- The GW-1B prediction extrapolates beyond the
  largest observed model and training token count.
- Raw-loss Huber fitting should not be confused
  with the separate log-Huber scaling-fit analysis.
- The original preliminary script was reconstructed.
  Its full-data prediction was reproduced, but
  exact equivalence of every original implementation
  detail has not been independently verified.

## 9. Conclusion

Using all 12 completed primary RQ 2.2 runs,
the estimated GW-1B validation loss at 1.15B
parameters and 20B training tokens is approximately
**2.36–2.37**.

The Least Squares prediction is **2.368566**,
and the Huber prediction is **2.360957**.

These values are scaling-law estimates rather
than observed production validation losses.
The agreement between fitting methods is not
a statistical uncertainty interval.

## 10. Reproducibility

Files:

- val_loss_results.csv
- val_loss_scaling_fit.py
- val_loss_fit_results.json
- ilaha_val_loss_analysis.md

Dependencies:

- NumPy 2.5.4
- SciPy 1.18.1

Run from the repository root:

    python team2/rq22/results/val_loss_scaling_fit.py

The script generates val_loss_fit_results.json
containing fitted parameters, GW-1B predictions,
training RMSE, holdout errors, and sensitivity results.
