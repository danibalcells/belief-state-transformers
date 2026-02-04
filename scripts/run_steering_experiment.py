from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from interventions.steering import SteeringIntervention

DEFAULT_SEQ_LEN = 10

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
    args = parser.parse_args()
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
    )


def _plot_steering(metrics: torch.Tensor, output_path: Path) -> None:
    if metrics.numel() == 0:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Steering", "Steering"])
        ax.set_ylabel("KL divergence")
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
    ax.scatter(x0, actual_no, color="red", alpha=0.25, s=8)
    ax.scatter(x1, actual_yes, color="red", alpha=0.25, s=8)
    ax.scatter(x0, counter_no, color="blue", alpha=0.25, s=8)
    ax.scatter(x1, counter_yes, color="blue", alpha=0.25, s=8)

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
        Line2D([0], [0], color="blue", lw=2, label="KL from optimal given counterfactual"),
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

    intervention = SteeringIntervention(
        model_checkpoint_path=args.model_checkpoint,
        steerable_type=args.steerable_type,
        steerable_checkpoint_path=args.steerable_checkpoint,
        layer=args.layer,
        device=args.device,
    )
    result = intervention.run(
        seq_len=args.seq_len, num_sequences=args.num_sequences, position=args.position
    )

    results_path = output_dir / "results.pt"
    torch.save({"metrics": result.metrics, "metadata": result.metadata}, results_path)

    config = {
        "model_checkpoint": str(args.model_checkpoint),
        "steerable_type": args.steerable_type,
        "steerable_checkpoint": str(args.steerable_checkpoint),
        "seq_len": args.seq_len,
        "num_sequences": args.num_sequences,
        "position": "last" if args.position == "last" else int(args.position),
        "layer": args.layer,
        "output_dir": str(output_dir),
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    image_dir = Path("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"steering_{timestamp}.png"
    _plot_steering(result.metrics, image_path)


if __name__ == "__main__":
    main()
