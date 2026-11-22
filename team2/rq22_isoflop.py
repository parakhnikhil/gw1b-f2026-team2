"""Compare adjacent RQ 2.2 model sizes at matched estimated compute."""
import argparse
import json
import math
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

SIZES = ("50m", "100m", "200m", "350m")


def read_curve(run):
    summary = json.loads((run / "summary.json").read_text())
    n_params = int(summary["n_params"])
    losses_by_compute = defaultdict(list)

    with (run / "metrics.jsonl").open() as log:
        for line in log:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("val/loss") is None or row.get("tokens_seen", 0) <= 0:
                continue
            compute = 6 * n_params * int(row["tokens_seen"])
            losses_by_compute[compute].append(float(row["val/loss"]))

    if len(losses_by_compute) < 2:
        raise ValueError(f"{run.name}: fewer than two distinct validation points")

    budgets = sorted(losses_by_compute)
    losses = [losses_by_compute[budget][-1] for budget in budgets]
    repeats = sum(len(values) - 1 for values in losses_by_compute.values())
    spread = max(max(values) - min(values) for values in losses_by_compute.values())
    print(
        f"{run.name}: {len(budgets)} points, "
        f"compute {budgets[0]:.3e} to {budgets[-1]:.3e}, "
        f"repeats {repeats}, max spread {spread:.6f}"
    )
    return budgets, losses


def loss_at(curve, compute):
    budgets, losses = curve
    index = bisect_left(budgets, compute)
    if budgets[index] == compute:
        return losses[index]

    left, right = budgets[index - 1], budgets[index]
    weight = math.log(compute / left) / math.log(right / left)
    return losses[index - 1] + weight * (losses[index] - losses[index - 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", type=Path, help="Directory containing the run folders")
    args = parser.parse_args()

    for horizon in (10, 20):
        print(f"\n=== t{horizon} ===")
        curves = {
            size: read_curve(args.runs_dir / f"team2-rq2.2-{size}-t{horizon}")
            for size in SIZES
        }
        print("\nMatched compute | smaller loss | larger loss | larger minus smaller")

        for smaller, larger in zip(SIZES, SIZES[1:]):
            small_curve, large_curve = curves[smaller], curves[larger]
            low = max(small_curve[0][0], large_curve[0][0])
            high = min(small_curve[0][-1], large_curve[0][-1])

            print(f"\n{smaller} vs {larger}")
            if low > high:
                print("No shared compute range")
                continue

            midpoint = min(high, max(low, math.sqrt(low * high)))
            for compute in sorted({low, midpoint, high}):
                small_loss = loss_at(small_curve, compute)
                large_loss = loss_at(large_curve, compute)
                print(
                    f"{compute:.3e} | {small_loss:.4f} | {large_loss:.4f} | "
                    f"{large_loss - small_loss:+.4f}"
                )


if __name__ == "__main__":
    main()
