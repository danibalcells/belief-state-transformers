from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from HMM import Mess3
from probes.base import SteerableProbe
from transformer import BeliefStateTransformer


class LinearProbe(SteerableProbe):
    def __init__(
        self,
        transformer: BeliefStateTransformer | None,
        layer: int | None = None,
        d_in: Optional[int] = None,
        d_out: Optional[int] = None,
        bias: bool = True,
    ) -> None:
        super().__init__(transformer=transformer, layer=layer)
        if d_in is None:
            if transformer is not None:
                d_in = transformer.cfg.d_model
            else:
                d_in = BeliefStateTransformer.from_paper_config(
                    vocab_size=Mess3().vocab_size
                ).cfg.d_model
        if d_out is None:
            d_out = Mess3().num_states
        self.linear = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def decode_from_belief(self, beliefs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("LinearProbe does not support decode_from_belief")

    def simplex(self, outputs: torch.Tensor) -> torch.Tensor:
        if outputs.shape[-1] != 3:
            raise ValueError(f"simplex projection requires output dim 3, got {outputs.shape[-1]}")
        u = outputs.new_tensor([-1.0, -1.0, 1.0])
        v = outputs.new_tensor([-1.0, 1.0, 0.0])
        x = (outputs * u).sum(dim=-1)
        y = (outputs * v).sum(dim=-1)
        return torch.stack([x, y], dim=-1)
