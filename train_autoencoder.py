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

from probe import Autoencoder
from utils.simplex import project_3d_to_simplex2d


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class AutoencoderTrainConfig:
    device: str
    dataset_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    train_fraction: float
    seed: int
    lambda_recon: float
    lambda_geometry: float
    output_dir: Optional[Path]


class AutoencoderTrainer:
    def __init__(self, config: AutoencoderTrainConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")

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

        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        self.eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

        self.model = Autoencoder(d_in=acts.shape[1], hidden_dim=2, bias=True).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.loss_fn = nn.MSELoss()

        wandb.init(
            project="belief-state-transformers",
            name=f"autoencoder_{self.run_id}",
            config={
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "train_fraction": config.train_fraction,
                "seed": config.seed,
                "dataset_path": str(config.dataset_path),
                "d_in": acts.shape[1],
                "hidden_dim": 2,
                "lambda_recon": config.lambda_recon,
                "lambda_geometry": config.lambda_geometry,
            },
        )

    def train(self) -> None:
        start_time = time.time()
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            train_recon = 0.0
            train_geom = 0.0
            train_total = 0.0
            train_count = 0
            for batch_acts, batch_beliefs in self.train_loader:
                batch_acts = batch_acts.to(self.device)
                batch_beliefs = batch_beliefs.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                encoded = self.model.encode(batch_acts)
                decoded = self.model.decode(encoded)
                recon_loss = self.loss_fn(decoded, batch_acts)
                target_simplex = project_3d_to_simplex2d(batch_beliefs)
                geom_loss = self.loss_fn(encoded, target_simplex)
                total_loss = (
                    self.config.lambda_recon * recon_loss
                    + self.config.lambda_geometry * geom_loss
                )
                total_loss.backward()
                self.optimizer.step()
                batch_size = batch_acts.shape[0]
                train_recon += float(recon_loss.item()) * batch_size
                train_geom += float(geom_loss.item()) * batch_size
                train_total += float(total_loss.item()) * batch_size
                train_count += batch_size

            self.model.eval()
            eval_recon = 0.0
            eval_geom = 0.0
            eval_total = 0.0
            eval_count = 0
            eval_latents: Optional[torch.Tensor] = None
            eval_targets: Optional[torch.Tensor] = None
            eval_labels: Optional[torch.Tensor] = None
            with torch.no_grad():
                for batch_acts, batch_beliefs in self.eval_loader:
                    batch_acts = batch_acts.to(self.device)
                    batch_beliefs = batch_beliefs.to(self.device)
                    encoded = self.model.encode(batch_acts)
                    decoded = self.model.decode(encoded)
                    recon_loss = self.loss_fn(decoded, batch_acts)
                    target_simplex = project_3d_to_simplex2d(batch_beliefs)
                    geom_loss = self.loss_fn(encoded, target_simplex)
                    total_loss = (
                        self.config.lambda_recon * recon_loss
                        + self.config.lambda_geometry * geom_loss
                    )
                    batch_size = batch_acts.shape[0]
                    eval_recon += float(recon_loss.item()) * batch_size
                    eval_geom += float(geom_loss.item()) * batch_size
                    eval_total += float(total_loss.item()) * batch_size
                    eval_count += batch_size
                    if eval_latents is None:
                        eval_latents = encoded.detach().to(device="cpu")
                        eval_targets = target_simplex.detach().to(device="cpu")
                        eval_labels = batch_beliefs.detach().argmax(dim=-1).to(device="cpu")

            avg_train_recon = train_recon / max(train_count, 1)
            avg_train_geom = train_geom / max(train_count, 1)
            avg_train_total = train_total / max(train_count, 1)
            avg_eval_recon = eval_recon / max(eval_count, 1)
            avg_eval_geom = eval_geom / max(eval_count, 1)
            avg_eval_total = eval_total / max(eval_count, 1)
            elapsed_s = time.time() - start_time
            print(
                "epoch="
                f"{epoch} train_total={avg_train_total:.6f} eval_total={avg_eval_total:.6f} "
                f"train_recon={avg_train_recon:.6f} eval_recon={avg_eval_recon:.6f} "
                f"train_geom={avg_train_geom:.6f} eval_geom={avg_eval_geom:.6f} "
                f"elapsed_s={elapsed_s:.2f}"
            )
            log_payload: dict[str, object] = {
                "train/total": avg_train_total,
                "train/recon": avg_train_recon,
                "train/geometry": avg_train_geom,
                "eval/total": avg_eval_total,
                "eval/recon": avg_eval_recon,
                "eval/geometry": avg_eval_geom,
            }
            if eval_latents is not None and eval_targets is not None and eval_labels is not None:
                coords_pred = eval_latents.numpy()
                coords_tgt = eval_targets.numpy()
                labels = eval_labels.numpy()

                fig, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
                axes[0].scatter(
                    coords_pred[:, 0], coords_pred[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8
                )
                axes[0].set_title("eval: latents")
                axes[0].set_aspect("equal", adjustable="box")
                axes[1].scatter(
                    coords_tgt[:, 0], coords_tgt[:, 1], c=labels, cmap="tab10", s=6, alpha=0.8
                )
                axes[1].set_title("eval: targets")
                axes[1].set_aspect("equal", adjustable="box")
                log_payload["simplex/eval"] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_payload, step=epoch)

        output_dir = self.config.output_dir or (Path("outputs") / "autoencoders" / self.run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "autoencoder.pt"
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "d_in": self.model.decoder.out_features,
                "hidden_dim": 2,
                "dataset_path": str(self.config.dataset_path),
            },
            output_path,
        )
        wandb.finish()


def parse_args() -> AutoencoderTrainConfig:
    parser = argparse.ArgumentParser(description="Train an autoencoder on residual activations.")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-geometry", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    return AutoencoderTrainConfig(
        device=args.device,
        dataset_path=Path(args.dataset_path),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
        seed=args.seed,
        lambda_recon=args.lambda_recon,
        lambda_geometry=args.lambda_geometry,
        output_dir=Path(args.output_dir) if args.output_dir is not None else None,
    )


def main() -> None:
    config = parse_args()
    trainer = AutoencoderTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
