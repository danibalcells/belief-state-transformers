from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from interventions.zero_ablation import ZeroAblationIntervention


@dataclass(frozen=True)
class SweepArgs:
    model_checkpoint: Path
    probe_checkpoint: Path
    dataset_path: Path
    mode: Literal["single", "all", "separate"]
    dimension: int | None
    step: float


def parse_args() -> SweepArgs:
    parser = argparse.ArgumentParser(description="Run zero-ablation lambda sweeps.")
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--probe-checkpoint", type=str, required=True)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["single", "all", "separate"], default="separate")
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--step", type=float, default=0.1)
    args = parser.parse_args()
    return SweepArgs(
        model_checkpoint=Path(args.model_checkpoint),
        probe_checkpoint=Path(args.probe_checkpoint),
        dataset_path=Path(args.dataset_path),
        mode=args.mode,
        dimension=args.dimension,
        step=args.step,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s:%(message)s")
    args = parse_args()
    intervention = ZeroAblationIntervention(
        model_checkpoint_path=args.model_checkpoint,
        probe_checkpoint_path=args.probe_checkpoint,
        dataset_path=args.dataset_path,
    )
    result = intervention.sweep_lambda(
        dimension=args.dimension, mode=args.mode, step=args.step
    )
    all_result = intervention.sweep_lambda(mode="all", step=args.step)
    print({"lambdas": result.lambdas, "mean_kls": result.mean_kls})
    lambdas = result.lambdas
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True, sharex=True, sharey=True)
    if isinstance(result.mean_kls[0], list):
        for idx, kls in enumerate(result.mean_kls):
            axes[idx].plot(lambdas, kls)
            axes[idx].set_title(f"dim {idx}")
    else:
        axes[0].plot(lambdas, result.mean_kls)
        axes[0].set_title("dim 0")
    axes[3].plot(lambdas, all_result.mean_kls)
    axes[3].set_title("all")
    axes[0].set_ylabel("KL")
    for axis in axes:
        axis.set_xlabel("lambda")
    output_dir = Path("images")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"zero_ablation_{timestamp}.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
