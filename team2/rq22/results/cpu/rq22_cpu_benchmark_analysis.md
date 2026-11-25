# RQ2.2 — CPU Downstream Benchmark Analysis

## 1. Objective

Evaluate downstream performance across four model sizes
(50M, 100M, 200M, and 350M) and three training token
budgets (10, 20, and 30 tokens per parameter).

The CPU results serve as a complete backup while the
A100 evaluations are pending.

## 2. Evaluation Setup

- Evaluation: zero-shot
- Device: CPU
- Batch size: 16
- Checkpoints: final training checkpoints
- Evaluation limit: None (full datasets)
- Total models: 12
- Metrics: accuracy and length-normalized accuracy

| Benchmark | Number of examples |
|---|---:|
| HellaSwag | 10,042 |
| ARC-Easy | 2,376 |
| PIQA | 1,838 |

## 3. Results

All scores below are percentages.

| Model | HellaSwag Acc | HellaSwag Norm | ARC-Easy Acc | ARC-Easy Norm | PIQA Acc | PIQA Norm |
|---|---:|---:|---:|---:|---:|---:|
| 50M-t10 | 27.01 | 27.40 | 43.48 | 38.89 | 58.43 | 57.83 |
| 50M-t20 | 27.45 | 28.38 | 47.81 | 42.76 | 59.96 | 59.68 |
| 50M-t30 | 27.29 | 29.04 | 48.40 | 43.22 | 59.96 | 59.74 |
| 100M-t10 | 27.76 | 29.86 | 50.59 | 44.87 | 60.94 | 59.90 |
| 100M-t20 | 28.80 | 31.05 | 52.74 | 45.66 | 62.95 | 61.48 |
| 100M-t30 | 29.08 | 32.43 | 53.37 | 47.39 | 62.35 | 61.81 |
| 200M-t10 | 29.75 | 32.98 | 54.38 | 48.86 | 63.38 | 63.38 |
| 200M-t20 | 31.31 | 35.66 | 56.78 | 50.55 | 65.23 | 64.64 |
| 200M-t30 | 32.11 | 37.03 | 58.29 | 51.73 | 66.92 | 65.83 |
| 350M-t10 | 32.10 | 37.65 | 57.70 | 50.97 | 66.21 | 65.89 |
| 350M-t20 | 33.89 | 40.73 | 61.32 | 55.30 | 66.43 | 67.19 |
| 350M-t30 | 35.03 | 42.84 | 61.49 | 56.78 | 67.63 | 68.28 |

## 4. Key Findings

### Effect of Model Size

At each fixed tokens-per-parameter budget, increasing
model size improves both accuracy and normalized
accuracy across all three benchmarks.

### Effect of Training Token Budget

Normalized accuracy increases consistently from t10
to t20 to t30 for every model size and benchmark.

Raw accuracy generally improves, with minor exceptions
such as HellaSwag for 50M and PIQA for 100M.

### Best Performing Configuration

The 350M-t30 model achieves the highest scores across
all six reported metrics.

Compared with 50M-t10, its normalized accuracy improves
by approximately:

- HellaSwag: 15.44 percentage points
- ARC-Easy: 17.89 percentage points
- PIQA: 10.45 percentage points

### Relationship with Validation Loss

The downstream results are consistent with the overall
validation loss trends: larger models and higher token
budgets generally achieve lower validation loss and
better downstream performance.

This is a descriptive observation rather than a formal
correlation or compute-optimality analysis.

## 5. Verification

All 12 result files were checked for:

- Correct run identifiers
- Final checkpoint steps
- Consistent batch sizes
- Full evaluation datasets
- CPU device metadata
- Both accuracy metrics

The consolidated CSV preserves the original numerical
precision of the JSON results.

## 6. Limitations and Next Steps

These results are CPU backup results. A100 evaluations
have been submitted but have not yet completed.

CPU and A100 scores will be compared once GPU evaluation
is complete. The highest-scoring configuration should
not automatically be interpreted as compute-optimal.

## 7. Result Files

- rq22_cpu_benchmark_results.csv
- Individual downstream.json files for all 12 models

The original per-example JSONL files remain on Pegasus.

