from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

BELIEF_SOURCE_TITLES: dict[str, str] = {
    "counterfactual": "Steering to counterfactual beliefs",
    "other_seq_reachable": "Steering to valid beliefs unreachable given sequence",
    "random_simplex": "Steering to random beliefs",
}


def _mean_and_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    n = values.size
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    z = 1.96 if confidence >= 0.95 else 2.576
    half = z * sem
    return mean, mean - half, mean + half


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot steering experiment metrics: mean ± CI.")
    parser.add_argument(
        "results_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to results.pt (e.g. outputs/experiments/steering_20260204_185012/results.pt)",
    )
    parser.add_argument("--output", "-o", type=Path, default=None)
    parser.add_argument("--confidence", type=float, default=0.95)
    args = parser.parse_args()

    if args.results_path is None:
        experiments = Path("outputs/experiments")
        dirs = sorted(experiments.glob("steering_*/results.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not dirs:
            raise SystemExit("No steering results found. Pass results_path (e.g. outputs/experiments/steering_XXX/results.pt).")
        results_path = dirs[0]
    else:
        results_path = Path(args.results_path)
    if not results_path.is_file():
        raise SystemExit(f"Not a file: {results_path}")

    data = torch.load(results_path, map_location="cpu", weights_only=False)
    metrics = data["metrics"]
    metadata = data.get("metadata", {})
    seq_idx = metadata.get("sequence_index")
    if isinstance(seq_idx, torch.Tensor):
        _, inverse = torch.unique(seq_idx, return_inverse=True)
        n = int(inverse.max().item()) + 1
        counts = torch.bincount(inverse, minlength=n).to(metrics.dtype)
        agg = torch.zeros(n, metrics.shape[1], dtype=metrics.dtype)
        for c in range(metrics.shape[1]):
            agg[:, c].scatter_add_(0, inverse, metrics[:, c])
        metrics = (agg / counts.unsqueeze(1)).numpy()
    else:
        metrics = metrics.numpy()
    if metrics.size == 0:
        raise SystemExit("No metrics in results.pt.")

    actual_no = metrics[:, 0]
    actual_steer = metrics[:, 1]
    counter_no = metrics[:, 2]
    counter_steer = metrics[:, 3]

    mean_actual_no, lo_actual_no, hi_actual_no = _mean_and_ci(actual_no, args.confidence)
    mean_actual_steer, lo_actual_steer, hi_actual_steer = _mean_and_ci(actual_steer, args.confidence)
    mean_counter_no, lo_counter_no, hi_counter_no = _mean_and_ci(counter_no, args.confidence)
    mean_counter_steer, lo_counter_steer, hi_counter_steer = _mean_and_ci(counter_steer, args.confidence)

    config_path = results_path.parent / "config.json"
    title = None
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        belief_source = config.get("belief_source")
        if belief_source is not None and belief_source in BELIEF_SOURCE_TITLES:
            title = BELIEF_SOURCE_TITLES[belief_source]

    x = np.array([0.0, 1.0])
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if title is not None:
        ax.set_title(title)

    mean_actual = np.array([mean_actual_no, mean_actual_steer])
    lo_actual = np.array([lo_actual_no, lo_actual_steer])
    hi_actual = np.array([hi_actual_no, hi_actual_steer])
    ax.fill_between(x, lo_actual, hi_actual, color="red", alpha=0.5)
    ax.plot(x, mean_actual, color="red", lw=1, marker="o", markersize=5, label="KL from optimal (actual)")
    ax.errorbar(x, mean_actual, yerr=[mean_actual - lo_actual, hi_actual - mean_actual], color="red", capsize=4, capthick=1.5, fmt="none", zorder=10)

    mean_counter = np.array([mean_counter_no, mean_counter_steer])
    lo_counter = np.array([lo_counter_no, lo_counter_steer])
    hi_counter = np.array([hi_counter_no, hi_counter_steer])
    ax.fill_between(x, lo_counter, hi_counter, color="blue", alpha=0.5)
    ax.plot(x, mean_counter, color="blue", lw=1, marker="o", markersize=5, label="KL from optimal (counterfactual)")
    ax.errorbar(x, mean_counter, yerr=[mean_counter - lo_counter, hi_counter - mean_counter], color="blue", capsize=4, capthick=1.5, fmt="none", zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(["No steering", "Steering"])
    ax.set_ylabel("KL divergence")
    ax.legend(loc="upper right")
    ax.set_ylim(bottom=0)

    out = args.output or Path("images") / f"steering_summary_{results_path.parent.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
