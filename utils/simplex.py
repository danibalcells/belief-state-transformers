from __future__ import annotations

import torch


def project_3d_to_simplex2d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != 3:
        raise ValueError(f"simplex projection requires last dim 3, got {x.shape[-1]}")
    u = x.new_tensor([-1.0, -1.0, 1.0])
    v = x.new_tensor([-1.0, 1.0, 0.0])
    x_coord = (x * u).sum(dim=-1)
    y_coord = (x * v).sum(dim=-1)
    coords = torch.stack([x_coord, y_coord], dim=-1)
    return (coords + 1.0) / 2.0
