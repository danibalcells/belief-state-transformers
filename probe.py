from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from HMM import Mess3
from transformer import BeliefStateTransformer
from utils.simplex import project_3d_to_simplex2d


class LinearProbe(nn.Module):
    def __init__(
        self,
        d_in: Optional[int] = None,
        d_out: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_in is None:
            d_in = BeliefStateTransformer.from_paper_config(
                vocab_size=Mess3().vocab_size
            ).cfg.d_model
        if d_out is None:
            d_out = Mess3().num_states
        self.linear = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def simplex(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.shape[-1] != 3:
            raise ValueError(f"simplex projection requires output dim 3, got {outputs.shape[-1]}")
        u = outputs.new_tensor([-1.0, -1.0, 1.0])
        v = outputs.new_tensor([-1.0, 1.0, 0.0])
        x = (outputs * u).sum(dim=-1)
        y = (outputs * v).sum(dim=-1)
        return torch.stack([x, y], dim=-1)


class Autoencoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        hidden_dim: int = 2,
        bias: bool = True,
        use_activation: bool = True,
    ) -> None:
        super().__init__()
        if use_activation:
            self.encoder = nn.Sequential(
                nn.Linear(d_in, hidden_dim, bias=bias),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Linear(d_in, hidden_dim, bias=bias)
        self.decoder = nn.Linear(hidden_dim, d_in, bias=bias)

    @classmethod
    def from_transformer(
        cls,
        transformer: BeliefStateTransformer,
        hidden_dim: int = 2,
        bias: bool = True,
        use_activation: bool = True,
    ) -> "Autoencoder":
        return cls(
            d_in=transformer.cfg.d_model,
            hidden_dim=hidden_dim,
            bias=bias,
            use_activation=use_activation,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def decode_from_belief(self, beliefs: torch.Tensor) -> torch.Tensor:
        latents = project_3d_to_simplex2d(beliefs)
        return self.decode(latents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
