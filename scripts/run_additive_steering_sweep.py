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
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from interventions.steering import AdditiveSteeringIntervention
from scripts.run_steering_experiment import _aggregate_metrics_by_sequence

DEFAULT_SEQ_LEN = 10


@dataclass(frozen=True)
class AdditiveSweepArgs:
    model_checkpoint: Path
    steerable_type: Literal["autoencoder", "vae"]
    steerable_checkpoint: Path
    num_sequences: int
    position: int | Literal["last"]
    layer: int | None
    output_dir: Path | None
    device: str | None
    seq_len: int = DEFAULT_SEQ_LEN
    step: float = 0.25


def _parse_position(value: str) -> int | Literal["last"]:
    if value == "last":
        return "last"
    return int(value)


def _lambdas_for_step(step: float) -> list[float]:
    n = int(round(1.0 / step))
    return list(np.linspace(0.0, 1.0, num=n + 1))


def parse_args() -> AdditiveSweepArgs:
    parser = argparse.ArgumentParser(
        description="Run additive steering across multiple lambda values."
    )
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument(
        "--steerable-type", type=str, choices=["autoencoder", "vae"], required=True
    )
    parser.add_argument("--steerable-checkpoint", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--num-sequences", type=int, default=10000)
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--step",
        type=float,
        default=0.25,
        help="Lambda step; 0 and 1 are always included (e.g. 0.25 -> 0, 0.25, 0.5, 0.75, 1.0)",
    )
    args = parser.parse_args()
    if args.step <= 0 or args.step > 1:
        parser.error("--step must be in (0, 1]")
    return AdditiveSweepArgs(
        model_checkpoint=Path(args.model_checkpoint),
        steerable_type=args.steerable_type,
        steerable_checkpoint=Path(args.steerable_checkpoint),
        seq_len=args.seq_len,
        num_sequences=args.num_sequences,
        position=_parse_position(args.position),
        layer=args.layer,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
        device=args.device,
        step=args.step,
    )


def _plot_sweep(
    lambdas: list[float],
    kl_actual_per_lambda: list[torch.Tensor],
    kl_counter_per_lambda: list[torch.Tensor],
    output_path: Path,
) -> None:
    lambdas_arr = np.array(lambdas)
    mean_actual = np.array(
        [t.mean().item() if t.numel() > 0 else float("nan") for t in kl_actual_per_lambda]
    )
    mean_counter = np.array(
        [t.mean().item() if t.numel() > 0 else float("nan") for t in kl_counter_per_lambda]
    )
    if np.all(np.isnan(mean_actual)) and np.all(np.isnan(mean_counter)):
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.set_xlabel("lambda")
        ax.set_ylabel("KL divergence")
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(lambdas_arr, mean_actual, "o-", color="red", label="KL(optimal actual || steered)")
    ax.plot(lambdas_arr, mean_counter, "s-", color="blue", label="KL(optimal counter || steered)")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean KL divergence")
    ax.legend(loc="upper right")
    ax.set_xticks(lambdas_arr)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(name)s:%(levelname)s:%(message)s")
    args = parse_args()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        Path("outputs") / "experiments" / f"additive_sweep_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    lambdas = _lambdas_for_step(args.step)
    kl_actual_per_lambda: list[torch.Tensor] = []
    kl_counter_per_lambda: list[torch.Tensor] = []

    for lam in tqdm(lambdas, desc="lambda"):
        logging.info("Running additive steering with lambda=%.4f", lam)
        intervention = AdditiveSteeringIntervention(
            model_checkpoint_path=args.model_checkpoint,
            steerable_type=args.steerable_type,
            steerable_checkpoint_path=args.steerable_checkpoint,
            lambda_=lam,
            layer=args.layer,
            device=args.device,
        )
        result = intervention.run(
            seq_len=args.seq_len,
            num_sequences=args.num_sequences,
            position=args.position,
        )
        metrics = result.metrics
        seq_idx = result.metadata.get("sequence_index")
        if isinstance(seq_idx, torch.Tensor) and metrics.numel() > 0:
            metrics = _aggregate_metrics_by_sequence(metrics.cpu(), seq_idx.cpu())
        if metrics.numel() > 0:
            kl_actual_per_lambda.append(metrics[:, 0].detach().cpu())
            kl_counter_per_lambda.append(metrics[:, 1].detach().cpu())
        else:
            kl_actual_per_lambda.append(torch.tensor([], dtype=torch.float64))
            kl_counter_per_lambda.append(torch.tensor([], dtype=torch.float64))

    results_path = output_dir / "results.pt"
    torch.save(
        {
            "lambdas": lambdas,
            "kl_actual_per_lambda": kl_actual_per_lambda,
            "kl_counter_per_lambda": kl_counter_per_lambda,
        },
        results_path,
    )
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
        "step": args.step,
        "lambdas": lambdas,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True))

    image_dir = Path("images")
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"additive_sweep_{timestamp}.png"
    _plot_sweep(
        lambdas,
        kl_actual_per_lambda,
        kl_counter_per_lambda,
        image_path,
    )


if __name__ == "__main__":
    main()
