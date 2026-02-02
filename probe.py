from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from HMM import Mess3
from transformer import BeliefStateTransformer


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
