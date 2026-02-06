from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from interventions.steering import (
    AdditiveSteeringIntervention,
    BeliefSource,
    SteeringIntervention,
)

DEFAULT_SEQ_LEN = 10

BELIEF_SOURCE_TITLES: dict[BeliefSource, str] = {
    "counterfactual": "Steering to counterfactual beliefs",
    "other_seq_reachable": "Steering to valid beliefs unreachable given sequence",
    "random_simplex": "Steering to random beliefs",
}

@dataclass(frozen=True)
class SteeringArgs:
    model_checkpoint: Path
    steerable_type: Literal["autoencoder", "vae"]
    steerable_checkpoint: Path
    num_sequences: int
    position: int | Literal["last"]
    layer: int | None
    output_dir: Path | None
    device: str | None
    seq_len: int = DEFAULT_SEQ_LEN
    additive: bool = False
    lambda_: float | None = None
    belief_source: BeliefSource = "counterfactual"


def _parse_position(value: str) -> int | Literal["last"]:
    if value == "last":
        return "last"
    return int(value)


def parse_args() -> SteeringArgs:
    parser = argparse.ArgumentParser(description="Run the steering experiment.")
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--steerable-type", type=str, choices=["autoencoder", "vae"], required=True)
    parser.add_argument("--steerable-checkpoint", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num-sequences", type=int, default=10000)
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--additive", action="store_true")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=None)
    parser.add_argument(
        "--belief-source",
        type=str,
        choices=["counterfactual", "other_seq_reachable", "random_simplex"],
        default="counterfactual",
    )
    args = parser.parse_args()
    if args.additive and args.lambda_ is None:
        parser.error("--lambda is required when --additive is set")
    if args.additive and args.lambda_ is not None and (args.lambda_ < 0 or args.lambda_ > 1):
        parser.error("--lambda must be in [0, 1] when --additive is set")
    return SteeringArgs(
        model_checkpoint=Path(args.model_checkpoint),
        steerable_type=args.steerable_type,
        steerable_checkpoint=Path(args.steerable_checkpoint),
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        position=_parse_position(args.position),
        layer=args.layer,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        device=args.device,
        additive=args.additive,
        lambda_=args.lambda_,
        belief_source=cast(BeliefSource, args.belief_source),
    )


def _plot_additive_steering(
    metrics: torch.Tensor,
    metadata: dict,
    output_path: Path,
    title: str | None = None,
) -> None:
    seq_idx = metadata.get("sequence_index")
    if isinstance(seq_idx, torch.Tensor):
        metrics = _aggregate_metrics_by_sequence(metrics.detach().cpu(), seq_idx.cpu())

    if metrics.numel() == 0:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_ylabel("KL divergence")
        if title is not None:
            ax.set_title(title)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    values = metrics.detach().cpu().numpy()
    kl_actual = values[:, 0]
    kl_counter = values[:, 1]

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    if title is not None:
        ax.set_title(title)
    ax.scatter(kl_actual, kl_counter, alpha=0.25, s=8)
    ax.set_xlabel("KL(optimal actual || steered pred)")
    ax.set_ylabel("KL(optimal injected || steered pred)")
    ax.axline((0, 0), slope=1, color="gray", linestyle="--", alpha=0.5)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _aggregate_metrics_by_sequence(metrics: torch.Tensor, sequence_index: torch.Tensor) -> torch.Tensor:
    _, inverse = torch.unique(sequence_index, return_inverse=True)
    n = inverse.max().item() + 1
    counts = torch.bincount(inverse, minlength=n).to(metrics.dtype)
    agg = torch.zeros(n, metrics.shape[1], dtype=metrics.dtype, device=metrics.device)
    for c in range(metrics.shape[1]):
        agg[:, c].scatter_add_(0, inverse, metrics[:, c])
    agg /= counts.unsqueeze(1)
    return agg


def _plot_steering(
    metrics: torch.Tensor,
    metadata: dict,
    output_path: Path,
    title: str | None = None,
) -> None:
    seq_idx = metadata.get("sequence_index")
    if isinstance(seq_idx, torch.Tensor):
        metrics = _aggregate_metrics_by_sequence(metrics.detach().cpu(), seq_idx.cpu())

    if metrics.numel() == 0:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Steering", "Steering"])
        ax.set_ylabel("KL divergence")
        if title is not None:
            ax.set_title(title)
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    values = metrics.detach().cpu().numpy()
    actual_no = values[:, 0]
    actual_yes = values[:, 1]
    counter_no = values[:, 2]
    counter_yes = values[:, 3]
    x0 = np.zeros_like(actual_no)
    x1 = np.ones_like(actual_no)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    if title is not None:
        ax.set_title(title)
    ax.scatter(x0, actual_no, color="red", alpha=0.1, s=8)
    ax.scatter(x1, actual_yes, color="red", alpha=0.1, s=8)
    ax.scatter(x0, counter_no, color="blue", alpha=0.1, s=8)
    ax.scatter(x1, counter_yes, color="blue", alpha=0.1, s=8)

    actual_segments = np.stack([np.stack([x0, actual_no], axis=1), np.stack([x1, actual_yes], axis=1)], axis=1)
    counter_segments = np.stack(
        [np.stack([x0, counter_no], axis=1), np.stack([x1, counter_yes], axis=1)], axis=1
    )
    ax.add_collection(LineCollection(actual_segments, colors="red", linewidths=0.5, alpha=0.15))
    ax.add_collection(LineCollection(counter_segments, colors="blue", linewidths=0.5, alpha=0.15))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Steering", "Steering"])
    ax.set_ylabel("KL divergence")
    legend_handles = [
        Line2D([0], [0], color="red", lw=2, label="KL from optimal given actual sequence"),
        Line2D([0], [0], color="blue", lw=2, label="KL from optimal given injected belief"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s:%(message)s")
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (Path("outputs") / "experiments" / f"steering_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.additive:
        assert args.lambda_ is not None
        intervention = AdditiveSteeringIntervention(
            model_checkpoint_path=args.model_checkpoint,
            steerable_type=args.steerable_type,
            steerable_checkpoint_path=args.steerable_checkpoint,
            lambda_=args.lambda_,
            layer=args.layer,
            device=args.device,
        )
    else:
        intervention = SteeringIntervention(
            model_checkpoint_path=args.model_checkpoint,
            steerable_type=args.steerable_type,
            steerable_checkpoint_path=args.steerable_checkpoint,
            layer=args.layer,
            device=args.device,
        )
    result = intervention.run(
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        position=args.position,
        belief_source=args.belief_source,
    )

    results_path = output_dir / "results.pt"
    torch.save({"metrics": result.metrics, "metadata": result.metadata}, results_path)

    print(f"Saved results to {results_path}.")

    config = {
        "model_checkpoint": str(args.model_checkpoint),
        "steerable_type": args.steerable_type,
        "steerable_checkpoint": str(args.steerable_checkpoint),
        "seq_len": args.seq_len,
        "num_sequences": args.num_sequences,
        "position": "last" if args.position == "last" else int(args.position),
        "layer": args.layer,
        "output_dir": str(output_dir),
        "additive": args.additive,
        "lambda_": args.lambda_,
        "belief_source": args.belief_source,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    image_dir = Path("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"steering_{timestamp}.png"
    plot_title = BELIEF_SOURCE_TITLES[args.belief_source]
    if args.additive:
        _plot_additive_steering(
            result.metrics, result.metadata, image_path, title=plot_title
        )
    else:
        _plot_steering(
            result.metrics, result.metadata, image_path, title=plot_title
        )


if __name__ == "__main__":
    main()
