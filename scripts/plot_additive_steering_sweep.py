from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot additive steering sweep results (lambda vs KL)."
    )
    parser.add_argument(
        "results",
        type=str,
        help="Path to results.pt or to directory containing results.pt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output figure path (default: images/additive_sweep_plot.png). Use e.g. images/additive_sweep_XXX_unique.png to avoid overwriting.",
    )
    args = parser.parse_args()

    path = Path(args.results)
    if path.is_dir():
        path = path / "results.pt"
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    lambdas = data["lambdas"]
    kl_actual_per_lambda = data["kl_actual_per_lambda"]
    kl_counter_per_lambda = data["kl_counter_per_lambda"]

    lambdas_arr = np.array(lambdas)
    mean_actual = np.array(
        [t.mean().item() if t.numel() > 0 else float("nan") for t in kl_actual_per_lambda]
    )
    mean_counter = np.array(
        [t.mean().item() if t.numel() > 0 else float("nan") for t in kl_counter_per_lambda]
    )
    sem_actual = np.array(
        [
            t.std(unbiased=True).item() / (t.numel() ** 0.5)
            if t.numel() > 1
            else 0.0 if t.numel() == 1
            else float("nan")
            for t in kl_actual_per_lambda
        ]
    )
    sem_counter = np.array(
        [
            t.std(unbiased=True).item() / (t.numel() ** 0.5)
            if t.numel() > 1
            else 0.0 if t.numel() == 1
            else float("nan")
            for t in kl_counter_per_lambda
        ]
    )
    z = 1.96
    lo_actual = mean_actual - z * sem_actual
    hi_actual = mean_actual + z * sem_actual
    lo_counter = mean_counter - z * sem_counter
    hi_counter = mean_counter + z * sem_counter

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.fill_between(lambdas_arr, lo_actual, hi_actual, color="red", alpha=0.2)
    ax.fill_between(lambdas_arr, lo_counter, hi_counter, color="blue", alpha=0.2)
    ax.plot(lambdas_arr, mean_actual, "o-", color="red", label="KL(optimal actual || steered)")
    ax.plot(lambdas_arr, mean_counter, "s-", color="blue", label="KL(optimal counter || steered)")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean KL divergence (95% CI)")
    ax.legend(loc="upper right")
    ax.set_xticks(lambdas_arr)

    out = Path(args.output) if args.output else Path("images") / "additive_sweep_plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
