from __future__ import annotations

import argparse
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.plotting import plot_beliefs_by_position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot belief states in 2D simplex per sequence position."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Directory to save the figure (default: images)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file path (overrides --output-dir)")
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Max points per subplot (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_path: Path | None
    if args.output is not None:
        output_path = Path(args.output)
    else:
        output_path = Path(args.output_dir) / f"beliefs_by_position_{dataset_path.stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_beliefs_by_position(
        dataset_path,
        output_path=output_path,
        max_points_per_plot=args.max_points,
    )
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
