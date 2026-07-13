"""Compute Diff-In influence for a single CIFAR-10 training sample.

After running ``train_cifar.py`` you should have a populated
``checkpoints/`` directory containing ``manifest.json`` and several
``ckpt_step*.pt`` files.  This script will:

    1.  Re-build the same data loaders used during training.
    2.  Instantiate :class:`diff_in.DiffInEstimator` against the saved
        checkpoints.
    3.  Pick a single training sample (``--sample-index``) and compute:
            * I(z, V)  — influence on validation loss
            * I(z, z)  — self-influence
            * ||I_θ(z)||  — magnitude of the influence on parameters
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from diff_in import CheckpointSpec, DiffInEstimator
from diff_in.utils import flat_norm
from train_cifar import build_model, CIFAR_MEAN, CIFAR_STD, pick_device


# ---------------------------------------------------------------------------
def load_manifest(ckpt_dir: Path) -> dict:
    with open(ckpt_dir / "manifest.json") as f:
        return json.load(f)


def make_eval_transform() -> transforms.Compose:
    """Eval-style transform (no augmentation) — used for the target sample
    and the validation loader so the influence estimate isn't perturbed by
    random crops / flips."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


def make_train_proxy_transform() -> transforms.Compose:
    """We *do* want the random training augmentation when sampling the
    proxy batch ``B_t``, exactly as during training."""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    print(f"[diff-in] device = {device}")

    ckpt_dir = Path(args.ckpt_dir)
    manifest = load_manifest(ckpt_dir)
    if len(manifest["checkpoints"]) == 0:
        raise RuntimeError("manifest.json has no checkpoints — run train_cifar.py first.")

    checkpoints: List[CheckpointSpec] = [
        CheckpointSpec(path=c["path"], step=c["step"], lr=c["lr"])
        for c in manifest["checkpoints"]
    ]
    print(f"[diff-in] {len(checkpoints)} checkpoint(s) loaded from {ckpt_dir}")
    for c in checkpoints:
        print(f"           step={c.step:6d}  lr={c.lr:.5f}  path={Path(c.path).name}")

    # ----- Data --------------------------------------------------------------
    train_set_proxy = datasets.CIFAR10(args.data_root, train=True, download=True,
                                       transform=make_train_proxy_transform())
    train_set_eval = datasets.CIFAR10(args.data_root, train=True, download=True,
                                      transform=make_eval_transform())
    val_full = datasets.CIFAR10(args.data_root, train=False, download=True,
                                transform=make_eval_transform())

    # We use a *small* validation subset for the per-step ∇L(V, θ_t) call.
    val_subset = Subset(val_full, list(range(args.val_size)))
    val_loader = DataLoader(val_subset, batch_size=args.val_batch_size,
                            shuffle=False, num_workers=0)

    # Proxy training loader: random augmented batches, drop the last odd batch.
    proxy_loader = DataLoader(train_set_proxy, batch_size=args.proxy_batch_size,
                              shuffle=True, num_workers=0, drop_last=True)

    # ----- Pick the target sample z -----------------------------------------
    idx = args.sample_index
    if not (0 <= idx < len(train_set_eval)):
        raise ValueError(f"sample-index out of range: {idx}")
    x_z, y_z = train_set_eval[idx]
    x_z = x_z.unsqueeze(0)
    y_z = torch.tensor([y_z], dtype=torch.long)
    print(f"[diff-in] target sample index={idx}  label={int(y_z)}  "
          f"(class={val_full.classes[int(y_z)] if False else train_set_eval.classes[int(y_z)]})")

    # ----- Build estimator ---------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    estimator = DiffInEstimator(
        model_factory=build_model,
        checkpoints=checkpoints,
        train_loader=proxy_loader,
        dataset_size=manifest["dataset_size"],
        criterion=criterion,
        device=device,
        epsilon=args.epsilon,
        batches_per_checkpoint=args.batches_per_checkpoint,
    )

    # ----- Compute influences -----------------------------------------------
    print("[diff-in] computing influence on validation loss  I(z, V) ...")
    i_loss = estimator.influence_on_loss(x_z, y_z, val_loader)
    print(f"   I(z, V) = {i_loss:+.6e}")

    print("[diff-in] computing self-influence  I(z, z) ...")
    i_self = estimator.self_influence(x_z, y_z)
    print(f"   I(z, z) = {i_self:+.6e}")

    if args.compute_params:
        print("[diff-in] computing influence on parameters  I_theta(z) ...")
        i_params = estimator.influence_on_params(x_z, y_z)
        norm = float(flat_norm(i_params).item())
        per_param = sum(p.numel() for p in i_params)
        print(f"   ||I_theta(z)||_2 = {norm:+.6e}   (over {per_param} parameters)")

    print("[diff-in] done.")


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute Diff-In influence for one sample.")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--ckpt-dir", default="./checkpoints")
    p.add_argument("--sample-index", type=int, default=42,
                   help="Index into CIFAR-10 train set whose influence we want.")
    p.add_argument("--val-size", type=int, default=512,
                   help="Use the first N validation samples for I(z, V).")
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--proxy-batch-size", type=int, default=128,
                   help="Batch size used to approximate G^t_{B_t}.")
    p.add_argument("--batches-per-checkpoint", type=int, default=1,
                   help="How many random batches we average per checkpoint.")
    p.add_argument("--epsilon", type=float, default=1e-3,
                   help="Finite-difference step in HVP.")
    p.add_argument("--compute-params", action="store_true",
                   help="Also compute the (large) I_theta(z) parameter influence.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    main()
