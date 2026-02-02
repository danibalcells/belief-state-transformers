from __future__ import annotations

from typing import Any, Literal, Tuple, cast

import torch
from jaxtyping import Float, Int
from transformer_lens import HookedTransformer, HookedTransformerConfig


class BeliefStateTransformer(HookedTransformer):
    @classmethod
    def from_paper_config(
        cls,
        vocab_size: int,
        device: str | torch.device | None = None,
    ) -> "BeliefStateTransformer":
        config = HookedTransformerConfig(
            n_layers=4,
            d_model=64,
            n_ctx=10,
            d_head=8,
            n_heads=1,
            d_mlp=256,
            act_fn="relu",
            d_vocab=vocab_size,
            d_vocab_out=vocab_size,
            normalization_type="LN",
            attention_dir="causal",
            device=str(device) if device is not None else None,
        )
        return cls(config)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return super().forward(*args, **kwargs)

    def forward_tokens(
        self, tokens: Int[torch.Tensor, "batch pos"]
    ) -> Float[torch.Tensor, "batch pos d_vocab"]:
        return cast(
            Float[torch.Tensor, "batch pos d_vocab"],
            self.forward(tokens, return_type="logits"),
        )

    def forward_with_residuals(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        resid_stage: Literal["pre", "mid", "post"] = "post",
        layers: list[int] | None = None,
    ) -> Tuple[
        Float[torch.Tensor, "batch pos d_vocab"],
        Float[torch.Tensor, "layer batch pos d_model"],
    ]:
        logits, cache = self.run_with_cache(tokens, return_type="logits")
        logits = cast(torch.Tensor, logits)
        hook_suffix = {
            "pre": "hook_resid_pre",
            "mid": "hook_resid_mid",
            "post": "hook_resid_post",
        }[resid_stage]
        if layers is None:
            layers = [self.cfg.n_layers - 1]
        activations = torch.stack(
            [cache[f"blocks.{layer}.{hook_suffix}"] for layer in layers],
            dim=0,
        )
        return logits, activations
