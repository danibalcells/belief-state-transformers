from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import torch
from torch import nn

from transformer import BeliefStateTransformer


def _resolve_position(sequence: torch.Tensor, position: int | Literal["last"]) -> int:
    seq_len = int(sequence.shape[1])
    if position == "last":
        return seq_len - 1
    if position < 0 or position >= seq_len:
        raise ValueError(f"position must be in [0, {seq_len - 1}], got {position}")
    return int(position)


def _ensure_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim == 2:
        return tensor
    raise ValueError(f"tensor must have shape (seq_len,) or (batch, seq_len), got {tensor.shape}")


def _ensure_belief_batch(beliefs: torch.Tensor) -> torch.Tensor:
    if beliefs.ndim == 1:
        return beliefs.unsqueeze(0)
    if beliefs.ndim == 2:
        return beliefs
    raise ValueError(f"belief_vector must have shape (states,) or (batch, states), got {beliefs.shape}")


class SteerableProbe(nn.Module, ABC):
    def __init__(
        self, transformer: BeliefStateTransformer | None, layer: int | None = None
    ) -> None:
        super().__init__()
        self._transformer = transformer
        if layer is None and transformer is not None:
            self.layer = transformer.cfg.n_layers - 1
        else:
            self.layer = int(layer) if layer is not None else None

    @property
    def transformer(self) -> BeliefStateTransformer | None:
        return self._transformer

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        raise NotImplementedError

    @abstractmethod
    def decode_from_belief(self, beliefs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def steer_to_belief(
        self,
        belief_vector: torch.Tensor,
        sequence: torch.Tensor,
        position: int | Literal["last"] = "last",
    ) -> torch.Tensor:
        if self.transformer is None or self.layer is None:
            raise ValueError("transformer and layer must be set before steering")
        device = next(self.transformer.parameters()).device
        tokens = _ensure_batch(sequence).to(device)
        belief_batch = _ensure_belief_batch(belief_vector).to(device, dtype=torch.float32)
        if belief_batch.shape[0] == 1 and tokens.shape[0] > 1:
            belief_batch = belief_batch.expand(tokens.shape[0], -1)
        if belief_batch.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"belief batch size {belief_batch.shape[0]} does not match tokens batch {tokens.shape[0]}"
            )
        pos = _resolve_position(tokens, position)
        self.eval()
        with torch.no_grad():
            replacement = self.decode_from_belief(belief_batch)

            def hook_fn(acts: torch.Tensor, hook: object | None = None) -> torch.Tensor:
                updated = acts.clone()
                updated[:, pos, :] = replacement
                return updated

            logits = self.transformer.run_with_hooks(
                tokens,
                return_type="logits",
                fwd_hooks=[(f"blocks.{self.layer}.hook_resid_post", hook_fn)],
            )
            probs = torch.softmax(logits[:, pos, :], dim=-1)
        return probs
