from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from HMM import Mess3
from utils.simplex import project_3d_to_simplex2d


def _rot_90_ccw(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return -coords[:, 1], coords[:, 0]


def _count_distinct_beliefs(
    actual: torch.Tensor,
    counterfactual: torch.Tensor,
    decimals: int = 8,
) -> tuple[int, int, int]:
    a = actual.to(torch.float64).round(decimals=decimals)
    c = counterfactual.to(torch.float64).round(decimals=decimals)
    actual_unique = torch.unique(a, dim=0).shape[0]
    counter_unique = torch.unique(c, dim=0).shape[0]
    pairs = torch.cat([a, c], dim=-1)
    pairs_unique = torch.unique(pairs, dim=0).shape[0]
    return int(actual_unique), int(counter_unique), int(pairs_unique)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    log_p = torch.log(p.clamp_min(1e-12))
    log_q = torch.log(q.clamp_min(1e-12))
    return (p * (log_p - log_q)).sum(dim=-1)


def _aggregate_by_distinct_counterfactual(
    counterfactual: torch.Tensor,
    values: torch.Tensor,
    decimals: int = 8,
) -> torch.Tensor:
    c = counterfactual.to(torch.float64).round(decimals=decimals)
    unique_c, group_ids = torch.unique(c, dim=0, return_inverse=True)
    n = unique_c.shape[0]
    counts = torch.bincount(group_ids, minlength=n).to(values.dtype)
    sums = torch.zeros(n, dtype=values.dtype, device=values.device)
    sums.scatter_add_(0, group_ids, values)
    return sums / counts.clamp_min(1)

def _plot_kl_analysis(
    actual: torch.Tensor,
    counterfactual: torch.Tensor,
    sequence_index: torch.Tensor | None,
    output_path: Path,
) -> None:
    hmm = Mess3()
    actual_next = hmm.optimal_next_token_probs_from_beliefs(actual.to(torch.float64))
    counter_next = hmm.optimal_next_token_probs_from_beliefs(counterfactual.to(torch.float64))
    kl = _kl_divergence(actual_next, counter_next)
    kl_per_intervention = _aggregate_by_distinct_counterfactual(counterfactual, kl)
    kl_np = kl_per_intervention.numpy()
    print(f"KL(actual || counterfactual) optimal next-token (per distinct intervention): min={float(kl_per_intervention.min().item()):.6f}, max={float(kl_per_intervention.max().item()):.6f}, n={len(kl_np)}")
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(10, 4))
    ax_hist.hist(kl_np, bins=50, edgecolor="black", alpha=0.7)
    ax_hist.set_xlabel("KL(actual || counterfactual)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Distribution of optimal next-token KL (one per distinct counterfactual)")
    ax_scatter.scatter(range(len(kl_np)), kl_np, alpha=0.3, s=8)
    ax_scatter.set_xlabel("distinct intervention index")
    ax_scatter.set_ylabel("mean KL(actual || counterfactual)")
    ax_scatter.set_title("Mean KL per distinct counterfactual")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_beliefs(actual: torch.Tensor, counterfactual: torch.Tensor, output_path: Path) -> None:
    actual_2d = project_3d_to_simplex2d(actual.detach().cpu())
    c_rounded = counterfactual.to(torch.float64).round(decimals=8)
    unique_c = torch.unique(c_rounded, dim=0)
    counter_2d = project_3d_to_simplex2d(unique_c)
    ax_x, ax_y = _rot_90_ccw(actual_2d)
    cf_x, cf_y = _rot_90_ccw(counter_2d)
    fig, (ax_actual, ax_counter) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    ax_actual.scatter(ax_x.numpy(), ax_y.numpy(), alpha=0.3, s=5)
    ax_actual.set_xlabel("simplex x")
    ax_actual.set_ylabel("simplex y")
    ax_actual.set_title("Actual beliefs")
    ax_actual.set_aspect("equal", adjustable="box")
    ax_counter.scatter(cf_x.numpy(), cf_y.numpy(), alpha=0.3, s=5)
    ax_counter.set_xlabel("simplex x")
    ax_counter.set_ylabel("simplex y")
    ax_counter.set_title("Counterfactual beliefs (distinct only)")
    ax_counter.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print tensor shapes from a steering results.pt file.")
    parser.add_argument("results_path", type=Path, help="Path to results.pt")
    parser.add_argument("--output", "-o", type=Path, default=None, help="Save 2D belief plot to this path")
    parser.add_argument("--output-kl", type=Path, default=None, help="Save KL of optimal next-token probs (actual||counterfactual) plot to this path")
    args = parser.parse_args()

    data = torch.load(args.results_path, map_location="cpu", weights_only=False)
    if "metrics" in data:
        print(f"metrics: {data['metrics'].shape} ({data['metrics'].dtype})")
    for key, value in data.get("metadata", {}).items():
        if isinstance(value, torch.Tensor):
            print(f"metadata['{key}']: {value.shape} ({value.dtype})")
        else:
            print(f"metadata['{key}']: {type(value).__name__} (not a tensor)")

    metadata = data.get("metadata", {})
    actual = metadata.get("actual_belief")
    counterfactual = metadata.get("counterfactual_belief")
    has_beliefs = isinstance(actual, torch.Tensor) and isinstance(counterfactual, torch.Tensor)

    tokens = metadata.get("tokens")
    if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
        n_sequences = torch.unique(tokens, dim=0).shape[0]
        print(f"Distinct token sequences: {int(n_sequences)}")

    if has_beliefs:
        n_actual, n_counter, n_pairs = _count_distinct_beliefs(actual, counterfactual)
        print(f"Distinct actual beliefs: {n_actual}")
        print(f"Distinct counterfactual beliefs: {n_counter}")
        print(f"Distinct (actual, counterfactual) pairs: {n_pairs}")

    if args.output is not None:
        if has_beliefs:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            _plot_beliefs(actual, counterfactual, args.output)
            print(f"Saved belief plot to {args.output}.")
        else:
            raise SystemExit("Cannot plot: metadata missing 'actual_belief' or 'counterfactual_belief'.")

    if args.output_kl is not None:
        if has_beliefs:
            args.output_kl.parent.mkdir(parents=True, exist_ok=True)
            seq_idx = metadata.get("sequence_index")
            if isinstance(seq_idx, torch.Tensor):
                seq_idx = seq_idx.cpu()
            else:
                seq_idx = None
            _plot_kl_analysis(actual, counterfactual, seq_idx, args.output_kl)
            print(f"Saved KL analysis plot to {args.output_kl}.")
        else:
            raise SystemExit("Cannot plot KL: metadata missing 'actual_belief' or 'counterfactual_belief'.")


if __name__ == "__main__":
    main()
