from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from probe import LinearProbe


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ProbeTrainConfig:
    device: str
    dataset_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_fraction: float
    seed: int
    output_dir: Optional[Path]


def parse_args() -> ProbeTrainConfig:
    parser = argparse.ArgumentParser(description="Train a linear probe on residual activations.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    return ProbeTrainConfig(
        device=args.device,
        dataset_path=Path(args.dataset_path),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
    )


def main() -> None:
    config = parse_args()
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    dataset_obj = torch.load(config.dataset_path, map_location="cpu")
    acts = dataset_obj["acts"].to(dtype=torch.float32)
    beliefs = dataset_obj["beliefs"].to(dtype=torch.float32)

    if acts.shape[0] != beliefs.shape[0]:
        raise ValueError("acts and beliefs must have the same number of samples")

    num_samples = acts.shape[0]
    num_train = int(num_samples * config.train_fraction)
    perm = torch.randperm(num_samples)
    train_idx = perm[:num_train]
    eval_idx = perm[num_train:]

    train_dataset = TensorDataset(acts[train_idx], beliefs[train_idx])
    eval_dataset = TensorDataset(acts[eval_idx], beliefs[eval_idx])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    probe = LinearProbe(d_in=acts.shape[1], d_out=beliefs.shape[1], bias=True).to(device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    wandb.init(
        project="belief-state-transformers",
        name=f"probe_{run_id}",
        config={
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "train_fraction": config.train_fraction,
            "seed": config.seed,
            "dataset_path": str(config.dataset_path),
            "d_in": acts.shape[1],
            "d_out": beliefs.shape[1],
        },
    )

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        probe.train()
        train_loss = 0.0
        train_count = 0
        for batch_acts, batch_beliefs in train_loader:
            batch_acts = batch_acts.to(device)
            batch_beliefs = batch_beliefs.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = probe(batch_acts)
            loss = loss_fn(preds, batch_beliefs)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * batch_acts.shape[0]
            train_count += batch_acts.shape[0]

        probe.eval()
        eval_loss = 0.0
        eval_count = 0
        eval_simplex_preds: Optional[torch.Tensor] = None
        eval_simplex_targets: Optional[torch.Tensor] = None
        eval_labels: Optional[torch.Tensor] = None
        with torch.no_grad():
            for batch_acts, batch_beliefs in eval_loader:
                batch_acts = batch_acts.to(device)
                batch_beliefs = batch_beliefs.to(device)
                preds = probe(batch_acts)
                loss = loss_fn(preds, batch_beliefs)
                eval_loss += float(loss.item()) * batch_acts.shape[0]
                eval_count += batch_acts.shape[0]
                if eval_simplex_preds is None:
                    eval_simplex_preds = probe.simplex(preds).detach().to(device="cpu")
                    eval_simplex_targets = probe.simplex(batch_beliefs).detach().to(device="cpu")
                    eval_labels = batch_beliefs.detach().argmax(dim=-1).to(device="cpu")

        avg_train = train_loss / max(train_count, 1)
        avg_eval = eval_loss / max(eval_count, 1)
        elapsed_s = time.time() - start_time
        print(f"epoch={epoch} train_mse={avg_train:.6f} eval_mse={avg_eval:.6f} elapsed_s={elapsed_s:.2f}")
        log_payload: dict[str, object] = {
            "train/mse": avg_train,
            "eval/mse": avg_eval,
        }
        if eval_simplex_preds is not None and eval_simplex_targets is not None and eval_labels is not None:
            coords_pred = eval_simplex_preds.numpy()
            coords_tgt = eval_simplex_targets.numpy()
            labels = eval_labels.numpy()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            axes[0].scatter(coords_pred[:, 0], coords_pred[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
            axes[0].set_title("eval: probe outputs (projected)")
            axes[0].set_aspect("equal", adjustable="box")
            axes[1].scatter(coords_tgt[:, 0], coords_tgt[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8)
            axes[1].set_title("eval: targets (projected)")
            axes[1].set_aspect("equal", adjustable="box")
            log_payload["simplex/eval"] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_payload, step=epoch)

    output_dir = config.output_dir or (Path("outputs") / "probes" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "probe.pt"
    torch.save(
        {
            "state_dict": probe.state_dict(),
            "d_in": acts.shape[1],
            "d_out": beliefs.shape[1],
            "dataset_path": str(config.dataset_path),
        },
        output_path,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
