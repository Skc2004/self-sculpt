"""
Trainer — Clean training loop with mixed precision, gradient clipping,
per-epoch metrics logging, and gate history recording.
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.sparsity_loss import SparsityLoss
from src.training.lambda_scheduler import LambdaScheduler


class Trainer:
    """Training orchestrator for PrunableNet models.

    Args:
        model: The model to train (PrunableNet or BaselineNet).
        optimizer: PyTorch optimizer (e.g., AdamW).
        lr_scheduler: Learning rate scheduler (e.g., CosineAnnealingLR).
        lambda_sched: LambdaScheduler for sparsity penalty.
        device: Torch device ('cuda' or 'cpu').
        sparsity_loss: SparsityLoss instance (None for baseline training).
        writer: Optional TensorBoard SummaryWriter.
        checkpoint_dir: Directory for saving model checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler=None,
        lambda_sched: LambdaScheduler | None = None,
        device: str = "cpu",
        sparsity_loss: SparsityLoss | None = None,
        writer=None,
        checkpoint_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.lambda_sched = lambda_sched
        self.device = device
        self.sparsity_loss = sparsity_loss
        self.writer = writer
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision setup
        self.use_amp = device == "cuda" and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # History tracking
        self.history = {
            "train_loss": [],
            "cls_loss": [],
            "spar_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lambda": [],
            "sparsity": [],
            "lr": [],
        }
        self.gate_history = []  # For animation: list of numpy gate arrays per epoch
        self.best_val_acc = 0.0

    def train_epoch(self, loader, epoch: int) -> dict:
        """Train for one epoch.

        Args:
            loader: Training DataLoader.
            epoch: Current epoch number (0-indexed).

        Returns:
            Dict of training metrics for this epoch.
        """
        self.model.train()
        lam = self.lambda_sched.get_lambda(epoch) if self.lambda_sched else 0.0

        total_cls_loss = 0.0
        total_spar_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    logits = self.model(x)
                    cls_loss = F.cross_entropy(logits, y)
                    spar_loss = self.sparsity_loss(self.model) if self.sparsity_loss else torch.tensor(0.0)
                    loss = cls_loss + lam * spar_loss

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                cls_loss = F.cross_entropy(logits, y)
                spar_loss = self.sparsity_loss(self.model) if self.sparsity_loss else torch.tensor(0.0)
                loss = cls_loss + lam * spar_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_cls_loss += cls_loss.item()
            total_spar_loss += spar_loss.item() if isinstance(spar_loss, torch.Tensor) else spar_loss
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        # Record gate values for animation
        if hasattr(self.model, "get_all_gates"):
            with torch.no_grad():
                gates = self.model.get_all_gates().cpu().numpy()
                self.gate_history.append(gates.copy())

        n_batches = len(loader)
        sparsity = 0.0
        if hasattr(self.model, "get_sparsity_report"):
            sparsity = self.model.get_sparsity_report().get("overall", 0.0)

        metrics = {
            "cls_loss": total_cls_loss / n_batches,
            "spar_loss": total_spar_loss / n_batches,
            "train_loss": (total_cls_loss + lam * total_spar_loss) / n_batches,
            "train_acc": correct / total,
            "lambda": lam,
            "sparsity": sparsity,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        # TensorBoard logging
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"train/{k}", v, epoch)

        # Update history
        for k, v in metrics.items():
            if k in self.history:
                self.history[k].append(v)

        return metrics

    @torch.no_grad()
    def evaluate(self, loader, epoch: int = 0) -> dict:
        """Evaluate on validation/test set.

        Args:
            loader: Validation/Test DataLoader.
            epoch: Current epoch (for logging).

        Returns:
            Dict with val_acc and val_loss.
        """
        self.model.eval()
        correct = 0
        total = 0
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        metrics = {
            "val_acc": correct / total,
            "val_loss": total_loss / len(loader),
        }

        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(f"val/{k}", v, epoch)

        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        verbose: bool = True,
    ) -> dict:
        """Full training loop.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation/Test DataLoader.
            epochs: Number of training epochs.
            verbose: Print progress each epoch.

        Returns:
            Training history dict.
        """
        print(f"Training on {self.device} for {epochs} epochs")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        if self.lambda_sched:
            print(f"Lambda schedule: {self.lambda_sched}")
        print("-" * 70)

        for epoch in range(epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader, epoch)

            # Track val accuracy in history
            self.history["val_acc"].append(val_metrics["val_acc"])

            # Step LR scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # Save best model
            if val_metrics["val_acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["val_acc"]
                self._save_checkpoint("best_model.pth")

            elapsed = time.time() - t0

            if verbose:
                sparsity_str = f"  spar={train_metrics['sparsity']:.1%}" if train_metrics["sparsity"] > 0 else ""
                print(
                    f"Epoch {epoch+1:3d}/{epochs} "
                    f"| loss={train_metrics['train_loss']:.4f} "
                    f"| cls={train_metrics['cls_loss']:.4f} "
                    f"| acc={train_metrics['train_acc']:.3f} "
                    f"| val={val_metrics['val_acc']:.3f} "
                    f"| lambda={train_metrics['lambda']:.1e}"
                    f"{sparsity_str}"
                    f"  [{elapsed:.1f}s]"
                )

        # Save final model
        self._save_checkpoint("final_model.pth")
        print(f"\nBest validation accuracy: {self.best_val_acc:.4f}")

        return self.history

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)
