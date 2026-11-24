"""RQ2.2 validation loss scaling analysis."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent

with (ROOT / "val_loss_results.csv").open() as f:
    rows = list(csv.DictReader(f))

N = np.array([float(r["actual_parameters"]) / 1e6 for r in rows])
D = np.array([float(r["actual_training_tokens"]) / 1e9 for r in rows])
Y = np.array([float(r["val_loss"]) for r in rows])
SIZES = np.array([r["model"] for r in rows])

assert len(rows) == 12


def scaling_loss(params, n, d):
    e, a, alpha, b, beta = params
    return e + a * n**(-alpha) + b * d**(-beta)


def fit(n, d, y, method):
    best = None

    for e0 in [1.0, 1.5, 2.0, 2.3]:
        for alpha0 in [0.2, 0.4, 0.6]:
            for beta0 in [0.2, 0.4, 0.6]:
                result = least_squares(
                    lambda p: scaling_loss(p, n, d) - y,
                    x0=[e0, 5.0, alpha0, 1.0, beta0],
                    bounds=(
                        [0, 0, 0.01, 0, 0.01],
                        [5, 1000, 2, 1000, 2],
                    ),
                    loss=method,
                    f_scale=0.01,
                    max_nfev=20000,
                )

                if result.success and (
                    best is None or result.cost < best.cost
                ):
                    best = result

    if best is None:
        raise RuntimeError("No successful fit")

    return best.x


def prediction(params):
    return float(scaling_loss(params, 1150.0, 20.0))


results = {
    "target_parameters": 1150000000,
    "target_training_tokens": 20000000000,
    "num_runs": len(rows),
    "parameter_unit": "million",
    "token_unit": "billion",
    "huber_f_scale": 0.01,
    "methods": {},
}

for method in ["linear", "huber"]:
    params = fit(N, D, Y, method)
    fitted = scaling_loss(params, N, D)

    entry = {
        "parameters": dict(zip(
            ["E", "A", "alpha", "B", "beta"],
            params.tolist(),
        )),
        "predicted_val_loss": prediction(params),
        "training_rmse": float(
            np.sqrt(np.mean((fitted - Y)**2))
        ),
        "holdout_350M": [],
        "leave_one_size_out": {},
    }

    mask = SIZES != "350M"
    holdout_params = fit(
        N[mask], D[mask], Y[mask], method
    )

    holdout_predictions = scaling_loss(
        holdout_params, N[~mask], D[~mask]
    )

    for row, pred in zip(
        [r for r in rows if r["model"] == "350M"],
        holdout_predictions,
    ):
        actual = float(row["val_loss"])

        entry["holdout_350M"].append({
            "tokens_per_parameter": int(
                row["tokens_per_parameter"]
            ),
            "actual_loss": actual,
            "predicted_loss": float(pred),
            "absolute_percentage_error": float(
                abs(pred - actual) / actual * 100
            ),
        })

    entry["holdout_mean_percentage_error"] = float(
        np.mean([
            r["absolute_percentage_error"]
            for r in entry["holdout_350M"]
        ])
    )

    for size in ["50M", "100M", "200M", "350M"]:
        mask = SIZES != size
        subset_params = fit(
            N[mask], D[mask], Y[mask], method
        )

        entry["leave_one_size_out"][size] = {
            "num_runs": int(mask.sum()),
            "predicted_val_loss": prediction(subset_params),
        }

    results["methods"][method] = entry

with (ROOT / "val_loss_fit_results.json").open("w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
