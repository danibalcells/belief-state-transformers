from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Check if probe weight rows (64-dim vectors) are linearly independent.")
    parser.add_argument("probe_path", type=Path, help="Path to probe.pt (e.g. outputs/probes/20260203_100503/probe.pt)")
    args = parser.parse_args()

    ckpt = torch.load(args.probe_path, map_location="cpu", weights_only=True)
    w = ckpt["state_dict"]["linear.weight"]
    r = torch.linalg.matrix_rank(w).item()
    m, n = w.shape[0], w.shape[1]
    indep = r == min(m, n)
    print(f"weight shape: {w.shape}, rank: {r}, linearly independent: {indep}")


if __name__ == "__main__":
    main()
