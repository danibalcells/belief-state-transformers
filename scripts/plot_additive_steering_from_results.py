from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_steering_experiment import _plot_additive_steering


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot additive steering scatter from saved results.pt (aggregated by distinct sequence)."
    )
    parser.add_argument("results_path", type=Path, help="Path to results.pt from an additive steering run")
    parser.add_argument("--output", "-o", type=Path, required=True, help="Output image path (e.g. images/steering_XXX_unique.png)")
    args = parser.parse_args()

    if not args.results_path.is_file():
        raise SystemExit(f"Not a file: {args.results_path}")

    data = torch.load(args.results_path, map_location="cpu", weights_only=False)
    metrics = data["metrics"]
    metadata = data.get("metadata", {})
    if metrics.shape[1] != 2:
        raise SystemExit("Expected 2 metric columns (additive steering). Check results.pt.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_additive_steering(metrics, metadata, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
