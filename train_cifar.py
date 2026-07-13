"""Train ResNet-18 on CIFAR-10 and save ``m`` evenly-spaced checkpoints.

Each checkpoint stores everything :class:`diff_in.DiffInEstimator` needs at
load time: the model's ``state_dict``, the global training step ``t`` and the
learning rate ``η_t`` that was in effect immediately after that step.

The list of checkpoint metadata is also written to ``checkpoints/manifest.json``
so downstream scripts (see ``compute_influence_example.py``) don't have to
re-derive the schedule.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18


# ---------------------------------------------------------------------------
# Model factory (kept top-level so it is picklable & importable from
# compute_influence_example.py)
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 10) -> nn.Module:
    """A ResNet-18 adapted for CIFAR-10 (3x3 stem, no max-pool)."""
    model = resnet18(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_dataloaders(data_root: str, batch_size: int, num_workers: int = 2):
    train_tfm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tfm)
    test_set = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False,
                             num_workers=num_workers, pin_memory=False)
    return train_set, test_set, train_loader, test_loader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def pick_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evenly_spaced_steps(total_iters: int, m: int) -> List[int]:
    """Pick ``m`` integer steps roughly uniformly in ``[1, total_iters]``."""
    if m <= 0:
        return []
    return [max(1, int(round(total_iters * (i + 1) / m))) for i in range(m)]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    device = pick_device(args.device)
    print(f"[train] using device: {device}")

    train_set, test_set, train_loader, test_loader = get_dataloaders(
        args.data_root, args.batch_size, args.num_workers
    )
    print(f"[train] |train|={len(train_set)}  |test|={len(test_set)}  "
          f"iters/epoch={len(train_loader)}")

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0,
                          weight_decay=args.weight_decay)
    # Cosine schedule -> learning rate at each step is needed by Diff-In.
    total_iters = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    ckpt_steps = set(evenly_spaced_steps(total_iters, args.num_checkpoints))
    print(f"[train] checkpoint steps = {sorted(ckpt_steps)} (total_iters={total_iters})")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for inputs, targets in train_loader:
            step += 1
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                running_loss += loss.item() * inputs.size(0)
                running_correct += (logits.argmax(1) == targets).sum().item()
                running_total += inputs.size(0)

            if step in ckpt_steps:
                lr_now = optimizer.param_groups[0]["lr"]
                ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
                torch.save(model.state_dict(), ckpt_path)
                manifest.append({
                    "path": str(ckpt_path),
                    "step": int(step),
                    "lr": float(lr_now),
                    "epoch": int(epoch),
                })
                print(f"  [ckpt] saved {ckpt_path.name}  step={step}  lr={lr_now:.5f}")

        train_acc = running_correct / max(running_total, 1)
        train_loss = running_loss / max(running_total, 1)

        test_acc, test_loss = evaluate(model, test_loader, criterion, device)
        elapsed = time.time() - t0
        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f} | "
              f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f} | "
              f"elapsed={elapsed:.1f}s")

    # Always include the final state of the model as well (handy for sanity checks).
    final_path = ckpt_dir / "ckpt_final.pt"
    torch.save(model.state_dict(), final_path)

    with open(ckpt_dir / "manifest.json", "w") as f:
        json.dump({
            "dataset_size": len(train_set),
            "total_iters": total_iters,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "checkpoints": manifest,
            "final_path": str(final_path),
        }, f, indent=2)
    print(f"[train] wrote manifest -> {ckpt_dir / 'manifest.json'}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_correct / max(total, 1), total_loss / max(total, 1)


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CIFAR-10 for Diff-In demo.")
    p.add_argument("--data-root", default="./data", help="Where CIFAR-10 lives.")
    p.add_argument("--ckpt-dir", default="./checkpoints",
                   help="Where to save model checkpoints + manifest.json.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-checkpoints", type=int, default=5,
                   help="m in the paper; how many evenly-spaced ckpts to save.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
