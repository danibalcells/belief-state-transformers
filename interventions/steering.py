from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from interventions.base import BaseIntervention, BaseInterventionResult
from probes.autoencoder import Autoencoder
from probes.base import SteerableProbe
from probes.vae import VariationalAutoencoder
from transformer import BeliefStateTransformer


@dataclass(frozen=True)
class SteeringResult(BaseInterventionResult):
    metrics: torch.Tensor
    metadata: dict[str, torch.Tensor]


def _infer_use_activation(state_dict: dict[str, torch.Tensor]) -> bool:
    return any(key.startswith("encoder.0.") for key in state_dict.keys())


def _load_autoencoder(
    checkpoint_path: Path,
    transformer: BeliefStateTransformer,
    layer: int | None,
    device: torch.device,
) -> Autoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    d_in = int(checkpoint["d_in"])
    hidden_dim = int(checkpoint["hidden_dim"])
    use_activation = _infer_use_activation(state_dict)
    model = Autoencoder(
        transformer=transformer,
        layer=layer,
        d_in=d_in,
        hidden_dim=hidden_dim,
        bias=True,
        use_activation=use_activation,
    )
    if model.transformer is None:
        object.__setattr__(model, "_transformer", transformer)
    if model.layer is None:
        model.layer = layer
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _load_vae(
    checkpoint_path: Path,
    transformer: BeliefStateTransformer,
    layer: int | None,
    device: torch.device,
) -> VariationalAutoencoder:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"]
    d_in = int(checkpoint["d_in"])
    latent_dim = int(checkpoint["latent_dim"])
    use_activation = _infer_use_activation(state_dict)
    model = VariationalAutoencoder(
        transformer=transformer,
        layer=layer,
        d_in=d_in,
        latent_dim=latent_dim,
        bias=True,
        use_activation=use_activation,
    )
    if model.transformer is None:
        object.__setattr__(model, "_transformer", transformer)
    if model.layer is None:
        model.layer = layer
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def _resolve_position(seq_len: int, position: int | Literal["last"]) -> int:
    if position == "last":
        return seq_len - 1
    if position < 0 or position >= seq_len:
        raise ValueError(f"position must be in [0, {seq_len - 1}], got {position}")
    return int(position)


def _kl_divergence(p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    log_p = torch.log(p.clamp_min(1e-12))
    return (p * (log_p - log_q)).sum(dim=-1)


class SteeringIntervention(BaseIntervention):
    def __init__(
        self,
        model_checkpoint_path: str | Path,
        steerable_type: Literal["autoencoder", "vae"],
        steerable_checkpoint_path: str | Path,
        layer: int | None = None,
        fallback_batch_size: int = 128,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(
            model_checkpoint_path=model_checkpoint_path,
            probe_checkpoint_path=None,
            dataset_path=None,
            fallback_batch_size=fallback_batch_size,
            device=device,
        )
        self.layer = int(layer) if layer is not None else self.transformer.cfg.n_layers - 1
        self.steerable_type = steerable_type
        self.steerable_checkpoint_path = Path(steerable_checkpoint_path)
        self.steerable = self._load_steerable()

    def _load_steerable(self) -> SteerableProbe:
        if self.steerable_type == "autoencoder":
            return _load_autoencoder(
                self.steerable_checkpoint_path, self.transformer, self.layer, self.device
            )
        if self.steerable_type == "vae":
            return _load_vae(
                self.steerable_checkpoint_path, self.transformer, self.layer, self.device
            )
        raise ValueError(f"unsupported steerable type: {self.steerable_type}")

    def run(
        self,
        seq_len: int,
        num_sequences: int = 10000,
        position: int | Literal["last"] = "last",
    ) -> SteeringResult:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if num_sequences <= 0:
            raise ValueError(f"num_sequences must be positive, got {num_sequences}")
        tokens, _ = self.hmm.generate_batch(batch_size=num_sequences, seq_len=seq_len)
        tokens = tokens.to(self.device)
        beliefs = self.hmm.belief_states(tokens)
        print(f'{tokens.shape=}')
        print(f'{beliefs.shape=}')
        pos = _resolve_position(seq_len, position)
        with torch.no_grad():
            logits = self.transformer.forward_tokens(tokens)
            log_probs_no_steer = torch.log_softmax(logits[:, pos, :], dim=-1).to(
                dtype=torch.float64
            )

        metrics: list[torch.Tensor] = []
        seq_indices: list[int] = []
        positions: list[int] = []
        actual_tokens: list[int] = []
        counter_tokens: list[int] = []
        actual_beliefs: list[torch.Tensor] = []
        counter_beliefs: list[torch.Tensor] = []

        emit = self.hmm._t_x.to(device=self.device, dtype=torch.float64)

        for idx in range(num_sequences):

            eta_prev = beliefs[idx, pos-1, :].to(dtype=torch.float64) 
            eta_current = beliefs[idx, pos, :].to(dtype=torch.float64)

            current_token = int(tokens[idx, pos].item())
            emission_prob_prev = self.hmm.optimal_next_token_probs_from_beliefs(
                eta_prev.unsqueeze(0)).squeeze(0)
            legal_current_tokens = torch.nonzero(emission_prob_prev > 0.0, as_tuple=False).flatten()

            # If only legal token is the generated one, skip the batch
            if int(legal_current_tokens.numel()) <= 1:
                continue

            # Compute optimal next token probabilities
            optimal_actual = self.hmm.optimal_next_token_probs_from_beliefs(eta_current.unsqueeze(0)).squeeze(0)

            for token in legal_current_tokens.tolist(): 

                if int(token) == current_token:
                    continue

                t_x = emit[int(token)]
                numer = torch.matmul(eta_prev, t_x)
                denom = numer.sum()
                eta_tilde = numer / denom

                optimal_counter = self.hmm.optimal_next_token_probs_from_beliefs(
                    eta_tilde.unsqueeze(0)
                ).squeeze(0)
                pred_steer = self.steerable.steer_to_belief(
                    eta_tilde, tokens[idx], position=pos
                ).squeeze(0)
                log_pred_steer = torch.log(pred_steer.clamp_min(1e-12)).to(dtype=torch.float64)
                log_pred_no = log_probs_no_steer[idx]

                kl_actual_no = _kl_divergence(optimal_actual, log_pred_no)
                kl_actual_steer = _kl_divergence(optimal_actual, log_pred_steer)
                kl_counter_no = _kl_divergence(optimal_counter, log_pred_no)
                kl_counter_steer = _kl_divergence(optimal_counter, log_pred_steer)
                metrics.append(
                    torch.stack(
                        [kl_actual_no, kl_actual_steer, kl_counter_no, kl_counter_steer],
                        dim=0,
                    )
                )
                seq_indices.append(idx)
                positions.append(pos)
                actual_tokens.append(current_token)
                counter_tokens.append(int(token))
                actual_beliefs.append(eta_current.detach().cpu())
                counter_beliefs.append(eta_tilde.detach().cpu())

        if metrics:
            metrics_tensor = torch.stack(metrics, dim=0).to(dtype=torch.float64, device="cpu")
            actual_beliefs_tensor = torch.stack(actual_beliefs, dim=0)
            counter_beliefs_tensor = torch.stack(counter_beliefs, dim=0)
        else:
            metrics_tensor = torch.empty((0, 4), dtype=torch.float64)
            actual_beliefs_tensor = torch.empty((0, 3), dtype=torch.float64)
            counter_beliefs_tensor = torch.empty((0, 3), dtype=torch.float64)

        metadata = {
            "tokens": tokens.detach().cpu(),
            "sequence_index": torch.tensor(seq_indices, dtype=torch.long),
            "position": torch.tensor(positions, dtype=torch.long),
            "actual_token": torch.tensor(actual_tokens, dtype=torch.long),
            "counterfactual_token": torch.tensor(counter_tokens, dtype=torch.long),
            "actual_belief": actual_beliefs_tensor,
            "counterfactual_belief": counter_beliefs_tensor,
        }
        return SteeringResult(metrics=metrics_tensor, metadata=metadata)
