"""Compare fitting objectives on the 350M holdout and the 23B forecast."""
import csv
from pathlib import Path

import numpy as np

from rq22_fit import fit, predict


def observations(rows):
    return np.array([
        [float(row[key]) for key in ("n_params", "tokens", "val_loss")]
        for row in rows
    ]).T


def main():
    path = Path(__file__).resolve().parent / "rq22/results/completed_inputs_2026-11-22.csv"
    with path.open(newline="") as source:
        rows = [row for row in csv.DictReader(source) if "-seed" not in row["run"]]

    for label, prefix in (
        ("Original 50M", "team2-rq2.2-50m-t"),
        ("Alternate 576x9", "team2-rq2.2-50m-576x9-t"),
    ):
        ladder = [
            row for row in rows
            if row["run"].startswith(prefix)
            or not row["run"].startswith("team2-rq2.2-50m-")
        ]
        train = [row for row in ladder if "-350m-" not in row["run"]]
        holdout = [row for row in ladder if "-350m-" in row["run"]]
        n, d, observed = observations(holdout)

        print(f"\n{label} ({len(train)} fit runs, {len(holdout)} holdouts)")
        for objective in ("log_huber", "log_mse", "raw_mse"):
            validation_fit = fit(*observations(train), objective=objective)
            errors = predict(validation_fit.x, n, d) - observed
            rmse = np.sqrt(np.mean(errors**2))
            full_fit = fit(*observations(ladder), objective=objective)
            forecast = predict(full_fit.x, 1.15e9, 23e9)
            bound_hit = bool(np.any(validation_fit.active_mask))
            error_text = ", ".join(
                f"{row['run'].removeprefix('team2-rq2.2-')}: {error:+.4f}"
                for row, error in zip(holdout, errors)
            )
            print(
                f"{objective:10s}  {error_text}  RMSE: {rmse:.4f}"
                f"  23B prediction: {forecast:.4f}  Holdout bound: {bound_hit}"
            )


if __name__ == "__main__":
    main()
