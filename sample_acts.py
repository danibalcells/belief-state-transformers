from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
import torch

from HMM import Mess3
from transformer import BeliefStateTransformer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class SampleActsConfig:
    device: str
    checkpoint_path: Path
    batch_size: int
    num_sequences: int
    seq_len: Optional[int]
    resid_stage: Literal["pre", "mid", "post"]
    layers: Optional[list[int]]
    output_dir: Optional[Path]


def _parse_layers(layers: Optional[str]) -> Optional[list[int]]:
    if layers is None:
        return None
    if layers.strip() == "":
        return None
    return [int(item) for item in layers.split(",")]


def parse_args() -> SampleActsConfig:
    parser = argparse.ArgumentParser(
        description="Sample HMM sequences and save transformer residual activations."
    )
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-sequences", type=int, default=10_000)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--resid-stage", type=str, choices=["pre", "mid", "post"], default="post")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    resid_stage = args.resid_stage
    return SampleActsConfig(
        device=args.device,
        checkpoint_path=Path(args.checkpoint_path),
        batch_size=args.batch_size,
        num_sequences=args.num_sequences,
        seq_len=args.seq_len,
        resid_stage=resid_stage,
        layers=_parse_layers(args.layers),
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
    )


def _move_hmm_to_device(hmm: Mess3, device: torch.device) -> None:
    hmm._t_x = hmm._t_x.to(device)
    hmm._t = hmm._t.to(device)
    hmm._pi = hmm._pi.to(device)
    hmm._joint = hmm._joint.to(device)


def main() -> None:
    config = parse_args()
    device = torch.device(config.device)
    print(
        "sample_acts params",
        {
            "device": config.device,
            "checkpoint_path": str(config.checkpoint_path),
            "batch_size": config.batch_size,
            "num_sequences": config.num_sequences,
            "seq_len": config.seq_len,
            "resid_stage": config.resid_stage,
            "layers": config.layers,
            "output_dir": str(config.output_dir) if config.output_dir is not None else None,
        },
    )
    hmm = Mess3()
    _move_hmm_to_device(hmm, device)
    model = BeliefStateTransformer.from_paper_config(
        vocab_size=hmm.vocab_size,
        device=device,
    )
    state_dict = torch.load(config.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    seq_len = config.seq_len if config.seq_len is not None else model.cfg.n_ctx
    if seq_len != model.cfg.n_ctx:
        raise ValueError(
            f"seq_len must match model context length {model.cfg.n_ctx}, got {seq_len}"
        )

    layers = config.layers
    if layers is not None and len(layers) != 1:
        raise ValueError("sample_acts only supports a single layer")

    acts_list: list[torch.Tensor] = []
    states_list: list[torch.Tensor] = []
    beliefs_list: list[torch.Tensor] = []

    num_remaining = config.num_sequences
    batch_index = 0
    with torch.no_grad():
        while num_remaining > 0:
            batch_size = min(config.batch_size, num_remaining)
            tokens, hidden = hmm.generate_batch(batch_size=batch_size, seq_len=seq_len)
            tokens = tokens.to(device)
            hidden = hidden.to(device)
            beliefs = hmm.belief_states(tokens)
            _, activations = model.forward_with_residuals(
                tokens, resid_stage=config.resid_stage, layers=layers
            )
            acts = activations[0]
            print(
                "batch_shapes",
                {
                    "batch_index": batch_index,
                    "tokens": tuple(tokens.shape),
                    "hidden": tuple(hidden.shape),
                    "beliefs": tuple(beliefs.shape),
                    "activations": tuple(activations.shape),
                    "acts": tuple(acts.shape),
                },
            )
            acts = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32, device="cpu")
            states = hidden[:, 1:].reshape(-1).to(dtype=torch.long, device="cpu")
            beliefs = beliefs[:, 1:, :].reshape(-1, beliefs.shape[-1]).to(
                dtype=torch.float32, device="cpu"
            )
            print(
                "batch_flat_shapes",
                {
                    "batch_index": batch_index,
                    "acts": tuple(acts.shape),
                    "states": tuple(states.shape),
                    "beliefs": tuple(beliefs.shape),
                },
            )

            acts_list.append(acts)
            states_list.append(states)
            beliefs_list.append(beliefs)
            num_remaining -= batch_size
            batch_index += 1

    acts_all = torch.cat(acts_list, dim=0)
    states_all = torch.cat(states_list, dim=0)
    beliefs_all = torch.cat(beliefs_list, dim=0)
    print(
        "final_shapes",
        {
            "acts": tuple(acts_all.shape),
            "states": tuple(states_all.shape),
            "beliefs": tuple(beliefs_all.shape),
        },
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_dir or (Path("outputs") / "datasets" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset.pt"

    torch.save(
        {
            "acts": acts_all,
            "states": states_all,
            "beliefs": beliefs_all,
            "seq_len": seq_len,
            "resid_stage": config.resid_stage,
            "layers": layers if layers is not None else [model.cfg.n_layers - 1],
            "num_sequences": config.num_sequences,
        },
        output_path,
    )


if __name__ == "__main__":
    main()
