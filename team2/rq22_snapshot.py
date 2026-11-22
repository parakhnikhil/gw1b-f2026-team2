"""Reproduce the GW-1B proxy holdout and production-target predictions."""
import argparse
import csv

import numpy as np
from scipy.optimize import least_squares

from rq22_fit import fit, predict

PREFIX = "team2-rq2.2-"


def observations(rows):
    return np.array([
        [float(row[key]) for key in ("n_params", "tokens", "val_loss")]
        for row in rows
    ]).T


def holdout_fit(rows):
    n, d, loss = observations(rows)
    alpha = 0.34

    def residuals(params):
        e, a, b, beta = params
        return np.log(predict((e, a, b, alpha, beta), n, d)) - np.log(loss)

    result = least_squares(
        residuals, [2, 0.5, 0.5, 0.3],
        bounds=([1e-8] * 4, [20, 100, 100, 2]),
        loss="huber", f_scale=1e-3, max_nfev=5000,
    )
    if not result.success:
        raise RuntimeError(result.message)
    e, a, b, beta = result.x
    return e, a, b, alpha, beta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path")
    args = parser.parse_args()
    with open(args.csv_path, newline="") as source:
        rows = list(csv.DictReader(source))
    if len({row["run"] for row in rows}) != len(rows):
        parser.error("Input contains duplicate run names.")

    shared = [row for row in rows if row["run"].startswith(
        tuple(PREFIX + size for size in ("100m-", "200m-", "350m-"))
    )]
    held_out = [row for row in shared if row["run"].startswith(PREFIX + "200m-")]
    for shape in ("50m-t", "50m-576x9-t"):
        shape_rows = [row for row in rows if row["run"].startswith(PREFIX + shape)]
        baseline = [row for row in shared if row["run"].startswith(PREFIX + "100m-")
                    and "-seed" not in row["run"]]
        theta = holdout_fit(shape_rows + baseline)
        print(shape)
        for row, n, d, observed in zip(held_out, *observations(held_out)):
            estimate = predict(theta, n, d)
            print(f"  {row['run']}: observed {observed:.4f}, "
                  f"predicted {estimate:.4f}, error {estimate - observed:+.4f}")

        for include_seeds in (False, True):
            inputs = shape_rows + [
                row for row in shared if include_seeds or "-seed" not in row["run"]
            ]
            result = fit(*observations(inputs))
            for target in (20, 23):
                estimate = predict(result.x, 1.15e9, target * 1e9)
                print(f"  {target}B tokens, {len(inputs)} runs: {estimate:.4f}")


if __name__ == "__main__":
    main()
