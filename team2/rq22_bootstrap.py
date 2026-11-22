import csv
from collections import Counter
from pathlib import Path

import numpy as np

from rq22_fit import fit, predict

path = Path("team2/rq22/results/completed_inputs_2026-11-22.csv")
prefix = "team2-rq2.2-"
with path.open(newline="") as source:
    rows = {row["run"].removeprefix(prefix): row for row in csv.DictReader(source)}

variants = ("50m", "50m-576x9")
budgets = {
    "50m": (10, 20, 30),
    "100m": (10, 20, 30),
    "200m": (10, 20),
    "350m": (10, 20),
}
seeds = [
    float(rows[name]["val_loss"])
    for name in ("100m-t20", "100m-t20-seed42", "100m-t20-seed1024")
]

rng = np.random.default_rng(42)
n_draws = 300
paired_predictions = []
failures = 0
bound_hits = Counter()

for draw in range(n_draws):
    sampled_settings = [
        (size, int(budget))
        for size, options in budgets.items()
        for budget in rng.choice(options, size=len(options), replace=True)
    ]
    sampled_seed_loss = float(np.mean(rng.choice(seeds, size=len(seeds), replace=True)))
    estimates = []

    for variant in variants:
        sample = []
        for size, budget in sampled_settings:
            shape = variant if size == "50m" else size
            row = rows[f"{shape}-t{budget}"]
            loss = (
                sampled_seed_loss
                if (size, budget) == ("100m", 20)
                else float(row["val_loss"])
            )
            sample.append((float(row["n_params"]), float(row["tokens"]), loss))

        try:
            result = fit(*np.asarray(sample).T)
        except ValueError:
            failures += 1
            break

        bound_hits[variant] += int(np.any(result.active_mask))
        estimates.append(float(predict(result.x, 1.15e9, 23e9)))

    if len(estimates) == 2:
        paired_predictions.append(estimates)

    if (draw + 1) % 50 == 0:
        print(f"Processed {draw + 1}/{n_draws} draws")

if not paired_predictions:
    raise SystemExit("No paired bootstrap fits converged.")

original, alternate = np.asarray(paired_predictions).T
print(f"\nPaired draws retained: {len(paired_predictions)}/{n_draws}")
print(f"Failed draws: {failures}")
print(f"Fits that hit a parameter bound: {dict(bound_hits)}")
for label, values in (
    ("Original 50M", original),
    ("Alternate 576x9", alternate),
    ("Alternate minus original", alternate - original),
):
    low, median, high = np.percentile(values, [5, 50, 95])
    print(f"{label}: median {median:.4f}, 90% range [{low:.4f}, {high:.4f}]")
