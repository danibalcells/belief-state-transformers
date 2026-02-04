from __future__ import annotations

import torch
from torch import nn

from probes.base import SteerableProbe
from transformer import BeliefStateTransformer
from utils.simplex import project_3d_to_simplex2d


class Autoencoder(SteerableProbe):
    def __init__(
        self,
        transformer: BeliefStateTransformer | None,
        layer: int | None = None,
        d_in: int | None = None,
        hidden_dim: int = 2,
        bias: bool = True,
        use_activation: bool = True,
    ) -> None:
        super().__init__(transformer=transformer, layer=layer)
        resolved_d_in = d_in
        if resolved_d_in is None and transformer is not None:
            resolved_d_in = transformer.cfg.d_model
        if resolved_d_in is None:
            raise ValueError("d_in is required when transformer is None")
        self.d_in = int(resolved_d_in)
        self.hidden_dim = int(hidden_dim)
        if use_activation:
            self.encoder = nn.Sequential(
                nn.Linear(self.d_in, self.hidden_dim, bias=bias),
                nn.ReLU(),
            )
        else:
            self.encoder = nn.Linear(self.d_in, self.hidden_dim, bias=bias)
        self.decoder = nn.Linear(self.hidden_dim, self.d_in, bias=bias)

    @classmethod
    def from_transformer(
        cls,
        transformer: BeliefStateTransformer,
        hidden_dim: int = 2,
        bias: bool = True,
        use_activation: bool = True,
        layer: int | None = None,
    ) -> "Autoencoder":
        return cls(
            transformer=transformer,
            layer=layer,
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
