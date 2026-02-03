from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.plotting import belief_probe_comparison_plot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot comparison of optimal vs probe-predicted belief states."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset .pt file")
    parser.add_argument("--probe", type=str, required=True, help="Path to probe .pt file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Directory to save the comparison image (default: images)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: comparison-<timestamp>.png)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output or f"comparison-{time.strftime('%Y%m%d_%H%M%S')}.png"
    output_path = output_dir / output_name
    belief_probe_comparison_plot(
        dataset_path=Path(args.dataset),
        probe_path=Path(args.probe),
        output_path=output_path,
    )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
