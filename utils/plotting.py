from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

from HMM import Mess3
from probes.linear import LinearProbe
from utils.simplex import project_3d_to_simplex2d


def belief_probe_comparison_plot(
    dataset_path: Path | str,
    probe_path: Path | str,
    device: Optional[torch.device] = None,
    max_points: Optional[int] = None,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    dataset_path = Path(dataset_path)
    probe_path = Path(probe_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(dataset_path, map_location="cpu")
    acts = data["acts"].to(dtype=torch.float32)
    beliefs_stored = data["beliefs"].to(dtype=torch.float32)

    if "tokens" in data:
        tokens = data["tokens"]
        hmm = Mess3()
        optimal_beliefs = hmm.belief_states(tokens)
        optimal_beliefs = optimal_beliefs[:, 1:, :].reshape(-1, optimal_beliefs.shape[-1]).to(
            dtype=torch.float32
        )
    else:
        optimal_beliefs = beliefs_stored

    if optimal_beliefs.shape[0] != acts.shape[0]:
        raise ValueError(
            f"optimal beliefs and acts length mismatch: {optimal_beliefs.shape[0]} vs {acts.shape[0]}"
        )

    ckpt = torch.load(probe_path, map_location=device)
    probe = LinearProbe(
        transformer=None, d_in=ckpt["d_in"], d_out=ckpt["d_out"], bias=True
    ).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    with torch.no_grad():
        acts_dev = acts.to(device)
        predicted_beliefs = probe(acts_dev).cpu().to(dtype=torch.float32)

    if max_points is not None and optimal_beliefs.shape[0] > max_points:
        perm = torch.randperm(optimal_beliefs.shape[0], generator=torch.Generator().manual_seed(0))[
            :max_points
        ]
        optimal_beliefs = optimal_beliefs[perm]
        predicted_beliefs = predicted_beliefs[perm]

    optimal_2d = project_3d_to_simplex2d(optimal_beliefs)
    predicted_2d = project_3d_to_simplex2d(predicted_beliefs)
    rgb = torch.clamp(optimal_beliefs, 0.0, 1.0).numpy()

    def rot_90_ccw(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return -x[:, 1], x[:, 0]

    opt_x, opt_y = rot_90_ccw(optimal_2d)
    pred_x, pred_y = rot_90_ccw(predicted_2d)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
    axes[0].scatter(
        opt_x.numpy(),
        opt_y.numpy(),
        c=rgb,
        s=0.5,
        alpha=0.8,
    )
    axes[0].set_title("Optimal belief states (projected)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].scatter(
        pred_x.numpy(),
        pred_y.numpy(),
        c=rgb,
        s=0.5,
        alpha=0.8,
    )
    axes[1].set_title("Probe predicted belief states (projected)")
    axes[1].set_aspect("equal", adjustable="box")

    if output_path is not None:
        fig.savefig(output_path)
    return fig


def probe_outputs_3d_plot(
    dataset_path: Path | str,
    probe_path: Path | str,
    device: Optional[torch.device] = None,
    max_points: Optional[int] = None,
    output_path: Optional[Path | str] = None,
) -> plt.Figure:
    dataset_path = Path(dataset_path)
    probe_path = Path(probe_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(dataset_path, map_location="cpu")
    acts = data["acts"].to(dtype=torch.float32)

    ckpt = torch.load(probe_path, map_location=device)
    probe = LinearProbe(
        transformer=None, d_in=ckpt["d_in"], d_out=ckpt["d_out"], bias=True
    ).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    with torch.no_grad():
        acts_dev = acts.to(device)
        outputs = probe(acts_dev).cpu().to(dtype=torch.float32)

    if max_points is not None and outputs.shape[0] > max_points:
        perm = torch.randperm(outputs.shape[0], generator=torch.Generator().manual_seed(0))[
            :max_points
        ]
        outputs = outputs[perm]

    x, y, z = outputs[:, 0].numpy(), outputs[:, 1].numpy(), outputs[:, 2].numpy()
    rgb = torch.clamp(outputs, 0.0, 1.0).numpy()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=rgb, s=0.5, alpha=0.8)
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ax.set_zlabel("dim 2")
    ax.set_title("Linear probe outputs (3D)")

    if output_path is not None:
        fig.savefig(output_path)
    return fig


def probe_outputs_3d_four_views(
    dataset_path: Path | str,
    probe_path: Path | str,
    device: Optional[torch.device] = None,
    max_points: Optional[int] = None,
    output_path: Optional[Path | str] = None,
    views: Optional[list[tuple[float, float]]] = None,
) -> plt.Figure:
    dataset_path = Path(dataset_path)
    probe_path = Path(probe_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.load(dataset_path, map_location="cpu")
    acts = data["acts"].to(dtype=torch.float32)

    ckpt = torch.load(probe_path, map_location=device)
    probe = LinearProbe(
        transformer=None, d_in=ckpt["d_in"], d_out=ckpt["d_out"], bias=True
    ).to(device)
    probe.load_state_dict(ckpt["state_dict"])
    probe.eval()

    with torch.no_grad():
        acts_dev = acts.to(device)
        outputs = probe(acts_dev).cpu().to(dtype=torch.float32)

    if max_points is not None and outputs.shape[0] > max_points:
        perm = torch.randperm(outputs.shape[0], generator=torch.Generator().manual_seed(0))[
            :max_points
        ]
        outputs = outputs[perm]

    x, y, z = outputs[:, 0].numpy(), outputs[:, 1].numpy(), outputs[:, 2].numpy()
    rgb = torch.clamp(outputs, 0.0, 1.0).numpy()

    if views is None:
        isometric_elev = 35.264
        views = [
            (25.0, 0.0),
            (25.0, 90.0),
            (isometric_elev, 45.0),
            (25.0, 180.0),
        ]
    if len(views) != 4:
        raise ValueError("views must contain exactly 4 (elev, azim) pairs")

    fig = plt.figure(figsize=(10, 10))
    for i, (elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        ax.scatter(x, y, z, c=rgb, s=0.5, alpha=0.8)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("probe 1")
        ax.set_ylabel("probe 2")
        ax.set_zlabel("probe 3")

    if output_path is not None:
        fig.savefig(output_path)
    return fig
