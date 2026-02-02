from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import wandb
from torch import nn

from HMM import Mess3
from transformer import BeliefStateTransformer


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class TrainConfig:
    device: str
    batch_size: int
    seq_len: int
    epochs: int
    learning_rate: float
    log_interval: int
    seed: int
    save_path: Optional[Path]


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train transformer on Mess3 observations.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    save_path = Path(args.save_path) if args.save_path is not None else None
    return TrainConfig(
        device=args.device,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        log_interval=args.log_interval,
        seed=args.seed,
        save_path=save_path,
    )


def train_step(
    model: BeliefStateTransformer, tokens: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    logits = model.forward_tokens(tokens)
    vocab_size = logits.shape[-1]
    loss = loss_fn(
        logits[:, :-1, :].reshape(-1, vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    return loss


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path("model_checkpoints") / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hmm = Mess3()
    hmm._t_x = hmm._t_x.to(device)
    hmm._t = hmm._t.to(device)
    hmm._pi = hmm._pi.to(device)
    hmm._joint = hmm._joint.to(device)

    model = BeliefStateTransformer.from_paper_config(
        vocab_size=hmm.vocab_size,
        device=device,
    )
    if config.seq_len != model.cfg.n_ctx:
        raise ValueError(
            f"seq_len must match model context length {model.cfg.n_ctx}, got {config.seq_len}"
        )

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    wandb.init(
        project="belief-state-transformers",
        name=run_id,
        config={
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "model_config": {
                "n_layers": model.cfg.n_layers,
                "d_model": model.cfg.d_model,
                "n_ctx": model.cfg.n_ctx,
                "d_head": model.cfg.d_head,
                "n_heads": model.cfg.n_heads,
                "d_mlp": model.cfg.d_mlp,
                "act_fn": model.cfg.act_fn,
            },
        },
    )

    start_time = time.time()
    for step in range(1, config.epochs + 1):
        tokens, _ = hmm.generate_batch(batch_size=config.batch_size, seq_len=config.seq_len)
        tokens = tokens.to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = train_step(model, tokens, loss_fn)
        loss.backward()
        optimizer.step()
        wandb.log({"train_loss": float(loss.item())}, step=step)

        if step == 1 or step % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"step={step} loss={float(loss.item()):.6f} elapsed_s={elapsed:.2f}")
        if step % 1000 == 0:
            model.eval()
            with torch.no_grad():
                eval_tokens, _ = hmm.generate_batch(
                    batch_size=config.batch_size, seq_len=config.seq_len
                )
                eval_tokens = eval_tokens.to(device)
                eval_loss = train_step(model, eval_tokens, loss_fn)
            model.train()
            wandb.log({"eval_loss": float(eval_loss.item())}, step=step)
            torch.save(model.state_dict(), checkpoint_dir / f"step_{step}.pt")

    if config.save_path is not None:
        config.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), config.save_path)
    wandb.finish()


if __name__ == "__main__":
    main()
