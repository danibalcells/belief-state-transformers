from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.plotting import probe_outputs_3d_four_views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a 2x2 grid of 3D probe output snapshots from different angles."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--probe", type=str, required=True, help="Path to probe .pt file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Directory to save the image (default: images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: probe_3d_views-<timestamp>.png)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Max points to plot (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output or f"probe_3d_views-{time.strftime('%Y%m%d_%H%M%S')}.png"
    output_path = output_dir / output_name
    probe_outputs_3d_four_views(
        dataset_path=Path(args.dataset),
        probe_path=Path(args.probe),
        max_points=args.max_points,
        output_path=output_path,
    )
    print(f"Saved 2x2 3D views to {output_path}")


if __name__ == "__main__":
    main()
