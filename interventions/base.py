from __future__ import annotations

from pathlib import Path
import logging
from dataclasses import dataclass
from typing import Callable, Sequence, cast

import torch

from HMM import Mess3
from probe import LinearProbe
from transformer import BeliefStateTransformer

HookFn = Callable[[torch.Tensor, object], torch.Tensor]


@dataclass(frozen=True)
class BaseInterventionResult:
    pass


class BaseIntervention:
    def __init__(
        self,
        model_checkpoint_path: str | Path,
        probe_checkpoint_path: str | Path,
        dataset_path: str | Path,
        fallback_batch_size: int = 128,
        device: str | torch.device | None = None,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        if fallback_batch_size <= 0:
            raise ValueError(f"fallback_batch_size must be positive, got {fallback_batch_size}")
        self.fallback_batch_size = fallback_batch_size
        resolved_device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.device: torch.device = resolved_device
        self.dataset: dict[str, object] = torch.load(dataset_path, map_location="cpu")
        self.hmm: Mess3 = Mess3()
        self.hmm._t_x = self.hmm._t_x.to(self.device)
        self.hmm._t = self.hmm._t.to(self.device)
        self.hmm._pi = self.hmm._pi.to(self.device)
        self.hmm._joint = self.hmm._joint.to(self.device)

        model = BeliefStateTransformer.from_paper_config(
            vocab_size=self.hmm.vocab_size,
            device=self.device,
        )
        state_dict = torch.load(model_checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self.transformer: BeliefStateTransformer = model
        self.transfromers: BeliefStateTransformer = model

        probe_state = torch.load(probe_checkpoint_path, map_location="cpu")
        d_in = int(probe_state["d_in"])
        d_out = int(probe_state["d_out"])
        probe = LinearProbe(d_in=d_in, d_out=d_out, bias=True)
        probe.load_state_dict(probe_state["state_dict"])
        probe.to(self.device)
        probe.eval()
        self.linear_probe: LinearProbe = probe
        self.logger.info(
            "loaded_intervention",
            extra={
                "device": str(self.device),
                "dataset_keys": list(self.dataset.keys()),
                "probe_dims": self._probe_dims(),
            },
        )

    def _probe_dims(self) -> int:
        return int(self.linear_probe.linear.weight.shape[0])

    def _normalized_vectors(self, dims: Sequence[int]) -> torch.Tensor:
        weight = self.linear_probe.linear.weight.detach().to(device=self.device, dtype=torch.float32)
        vectors = weight[torch.tensor(list(dims), dtype=torch.long, device=self.device)]
        norms = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return vectors / norms

    def _lambda_grid(self, step: float) -> list[float]:
        if step <= 0.0 or step > 1.0:
            raise ValueError(f"step must be in (0, 1], got {step}")
        num_steps = int(round(1.0 / step))
        lambdas = [round(i * step, 10) for i in range(num_steps + 1)]
        if lambdas[-1] != 1.0:
            lambdas.append(1.0)
        return lambdas

    def _dataset_tokens(self) -> torch.Tensor:
        if "tokens" not in self.dataset:
            num_sequences = int(self.dataset["num_sequences"])
            seq_len = int(self.dataset["seq_len"])
            self.logger.warning(
                "tokens_missing_fallback_sampling",
                extra={
                    "num_sequences": num_sequences,
                    "seq_len": seq_len,
                    "fallback_batch_size": self.fallback_batch_size,
                    "note": "dataset missing tokens; sampling fresh HMM sequences",
                },
            )
            tokens_list: list[torch.Tensor] = []
            num_remaining = num_sequences
            while num_remaining > 0:
                batch_size = min(self.fallback_batch_size, num_remaining)
                batch_tokens, _ = self.hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)
                tokens_list.append(batch_tokens.to(device="cpu"))
                num_remaining -= batch_size
            self.dataset["tokens"] = torch.cat(tokens_list, dim=0)
        tokens = torch.as_tensor(self.dataset["tokens"])
        if tokens.ndim != 2:
            raise ValueError(f"dataset tokens must have shape (batch, seq_len), got {tuple(tokens.shape)}")
        return tokens.to(self.device)

    def _mean_kl_from_logits(self, logits: torch.Tensor, tokens: torch.Tensor) -> float:
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1).to(dtype=torch.float64)
        optimal_probs = self.hmm.optimal_next_token_probs(tokens)
        log_optimal = torch.log(optimal_probs.clamp_min(1e-12))
        kl = (optimal_probs * (log_optimal - log_probs)).sum(dim=-1)
        return float(kl.mean().item())

    def _accuracy_from_logits(self, logits: torch.Tensor, tokens: torch.Tensor) -> float:
        preds = logits[:, :-1, :].argmax(dim=-1)
        targets = tokens[:, 1:]
        correct = (preds == targets).to(dtype=torch.float32)
        return float(correct.mean().item())

    def _run_with_hooks(self, hooks: list[tuple[str, HookFn]]) -> torch.Tensor:
        tokens = self._dataset_tokens()
        with torch.no_grad():
            logits = self.transformer.run_with_hooks(
                tokens, return_type="logits", fwd_hooks=hooks
            )
        return torch.as_tensor(logits)

    def _dataset_hook_suffix(self) -> str:
        resid_stage = str(self.dataset["resid_stage"])
        return {"pre": "hook_resid_pre", "mid": "hook_resid_mid", "post": "hook_resid_post"}[
            resid_stage
        ]

    def _dataset_layers(self) -> list[int]:
        return cast(list[int], self.dataset["layers"])
