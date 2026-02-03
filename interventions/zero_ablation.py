from __future__ import annotations

from typing import Literal

import torch
from einops import einsum

from dataclasses import dataclass

from interventions.base import BaseIntervention, BaseInterventionResult, HookFn


@dataclass(frozen=True)
class ZeroAblationResult(BaseInterventionResult):
    lambdas: list[float]
    mean_kls: list[float] | list[list[float]]


class ZeroAblationIntervention(BaseIntervention):
    def zero_ablation(self, dimension: int, lambda_value: float) -> float:
        vectors = self._normalized_vectors([dimension])
        return self._run_intervention(vectors, lambda_value)

    def sweep_lambda(
        self,
        dimension: int | None = None,
        mode: Literal["single", "all", "separate"] = "single",
        step: float = 0.1,
    ) -> ZeroAblationResult:
        lambdas = self._lambda_grid(step)
        if mode == "single":
            if dimension is None:
                raise ValueError("dimension is required when mode='single'")
            mean_kls = [self.zero_ablation(dimension, lam) for lam in lambdas]
            return ZeroAblationResult(lambdas=lambdas, mean_kls=mean_kls)
        if mode == "all":
            vectors = self._normalized_vectors(list(range(self._probe_dims())))
            mean_kls = [self._run_intervention(vectors, lam) for lam in lambdas]
            return ZeroAblationResult(lambdas=lambdas, mean_kls=mean_kls)
        if mode == "separate":
            mean_kls = []
            for dim in range(self._probe_dims()):
                mean_kls.append([self.zero_ablation(dim, lam) for lam in lambdas])
            return ZeroAblationResult(lambdas=lambdas, mean_kls=mean_kls)
        raise ValueError(f"unsupported mode: {mode}")

    def _run_intervention(self, vectors: torch.Tensor, lambda_value: float) -> float:
        hook_suffix = self._dataset_hook_suffix()
        layers = self._dataset_layers()


        def hook_fn(acts: torch.Tensor, hook: object | None = None) -> torch.Tensor:
            dots = einsum(acts, vectors, "b p d, k d -> b p k")
            proj = einsum(dots, vectors, "b p k, k d -> b p d")
            return acts - (lambda_value * proj)

        hooks: list[tuple[str, HookFn]] = [
            (f"blocks.{int(layer)}.{hook_suffix}", hook_fn) for layer in layers
        ]
        logits = self._run_with_hooks(hooks)
        tokens = self._dataset_tokens()
        return self._mean_kl_from_logits(logits, tokens)
