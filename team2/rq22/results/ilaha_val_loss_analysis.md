# RQ 2.2 — GW-1B Validation Loss Analysis

## Objective

Estimate the validation loss of GW-1B (1.15B parameters, 20B training tokens) using the completed RQ 2.2 proxy training runs.

## Dataset

The analysis uses 10 completed primary runs across four model sizes.

| Model | Actual parameters | t10 loss | t20 loss | t30 loss |
|---|---:|---:|---:|---:|
| 50M | 50,213,376 | 3.277724 | 3.151757 | 3.094283 |
| 100M | 100,092,672 | 3.034312 | 2.932259 | 2.882846 |
| 200M | 200,573,952 | 2.803991 | 2.699287 | Pending |
| 350M | 353,614,080 | 2.667833 | 2.593308 | Pending |

Additional seeds and the 50M-576x9 architecture variant were excluded from the primary fit.

## Scaling Law

L(N, D) = E + A / N^alpha + B / D^beta

N is measured in millions of parameters, and D in billions of tokens.

## Fitting Results

| Metric | Least Squares | Huber |
|---|---:|---:|
| GW-1B predicted val/loss | 2.358094 | 2.358014 |
| 350M holdout mean error | 1.512% | 1.526% |
| 350M-t10 holdout error | 1.288% | 1.280% |
| 350M-t20 holdout error | 1.737% | 1.771% |

Least Squares training RMSE: 0.010527

### Least Squares Parameters

- E = 1.606482
- A = 2.196038
- alpha = 0.227277
- B = 0.649111
- beta = 0.247761

### Huber Parameters

- E = 1.429443
- A = 2.257390
- alpha = 0.166609
- B = 0.550669
- beta = 0.290167

## Leave-One-Model-Size-Out Sensitivity

| Excluded model size | Runs used | GW-1B predicted val/loss |
|---|---:|---:|
| None | 10 | 2.358094 |
| 50M | 7 | 2.444598 |
| 100M | 7 | 2.426130 |
| 200M | 8 | 2.346413 |
| 350M | 8 | 2.254232 |

The prediction varies from 2.254232 to 2.444598 depending on the excluded model size. This is a sensitivity range, not a statistical confidence interval.

## Interpretation

Least Squares and Huber produce nearly identical full-data predictions, although their fitted coefficients differ.

Holding out the 350M models gives errors of approximately 1.3% and 1.7%. However, the leave-one-size-out results show that extrapolation to GW-1B is sensitive to the available proxy model sizes.

## Preliminary Conclusion

The estimated validation loss for GW-1B at 1.15B parameters and 20B training tokens is approximately **2.3581**.

This is a preliminary estimate, not an observed production validation loss. The 200M-t30 and 350M-t30 runs were unavailable during this analysis.

The results were calculated using raw-loss nonlinear least squares and raw-loss Huber fitting. They should not be confused with the log-Huber fitting method used in the separate RQ 2.2 scaling-fit analysis.
EOF
