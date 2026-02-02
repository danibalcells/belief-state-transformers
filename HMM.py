from __future__ import annotations

from typing import Final
from typing import Tuple

import torch
from einops import rearrange


def _stationary_distribution(transition: torch.Tensor) -> torch.Tensor:
    if transition.ndim != 2 or transition.shape[0] != transition.shape[1]:
        raise ValueError(f"transition must be square, got shape={tuple(transition.shape)}")
    n: int = int(transition.shape[0])
    evals, evecs = torch.linalg.eig(transition.T.to(torch.float64))
    idx: int = int(torch.argmin(torch.abs(evals - torch.tensor(1.0, dtype=evals.dtype))).item())
    v = evecs[:, idx].real
    v = torch.clamp(v, min=0.0)
    if float(v.sum().item()) == 0.0:
        raise RuntimeError("failed to compute stationary distribution (all-zero eigenvector)")
    v = v / v.sum()
    if v.shape != (n,):
        raise RuntimeError(f"stationary distribution has wrong shape: {tuple(v.shape)}")
    return v


class Mess3:
    vocab: Final[tuple[str, str, str]] = ("A", "B", "C")

    def __init__(self) -> None:
        t_a = torch.tensor(
            [
                [0.765, 0.00375, 0.00375],
                [0.0425, 0.0675, 0.00375],
                [0.0425, 0.00375, 0.0675],
            ],
            dtype=torch.float64,
        )
        t_b = torch.tensor(
            [
                [0.0675, 0.0425, 0.00375],
                [0.00375, 0.765, 0.00375],
                [0.00375, 0.0425, 0.0675],
            ],
            dtype=torch.float64,
        )
        t_c = torch.tensor(
            [
                [0.0675, 0.00375, 0.0425],
                [0.00375, 0.0675, 0.0425],
                [0.00375, 0.00375, 0.765],
            ],
            dtype=torch.float64,
        )
        self._t_x: torch.Tensor = torch.stack([t_a, t_b, t_c], dim=0)
        self._t: torch.Tensor = t_a + t_b + t_c
        # We sample the initial hidden state from the stationary distribution π of T = Σx T(x),
        # matching the paper’s training setup and avoiding artificial “startup transients” from an
        # arbitrary initial state (i.e., s0 ~ π makes the process stationary from the first step).
        self._pi: torch.Tensor = _stationary_distribution(self._t)

        self._joint: torch.Tensor = rearrange(self._t_x, "x i j -> i (x j)").contiguous()

    @property
    def num_states(self) -> int:
        return 3

    @property
    def vocab_size(self) -> int:
        return 3

    def generate_batch(self, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        device = self._t_x.device
        # Initial hidden state s0 ~ π (stationary distribution over latent states).
        states = torch.multinomial(self._pi.to(device), num_samples=batch_size, replacement=True).to(
            torch.long
        )
        seq = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
        hidden = torch.empty((batch_size, seq_len + 1), dtype=torch.long, device=device)
        hidden[:, 0] = states

        for t in range(seq_len):
            probs = self._joint.index_select(0, states).to(device)
            idx = rearrange(torch.multinomial(probs, num_samples=1, replacement=True), "b 1 -> b")
            emission = torch.div(idx, 3, rounding_mode="floor")
            next_state = idx.remainder(3)
            seq[:, t] = emission.to(torch.long)
            states = next_state.to(torch.long)
            hidden[:, t + 1] = states

        return seq, hidden

