"""Low-level helpers used by :class:`diff_in.estimator.DiffInEstimator`.

This module deliberately stays *small* and *stateless* so that the high-level
estimator in :mod:`diff_in.estimator` reads close to the math in the paper
("Understanding Data Influence with Differential Approximation").

The two ideas we need:

1.  Compute parameter gradients of a loss as a flat list of tensors that has
    exactly the same structure as ``list(model.parameters())``.
2.  Approximate a Hessian-vector product ``H @ v`` via the classic
    finite-difference rule (Pearlmutter, 1994; Eq.(6) of the paper)::

            H v  ≈  ( ∇L(θ + ε v) − ∇L(θ) ) / ε

    which only requires *first-order* gradient calls.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, Iterator, List, Sequence

import torch
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Parameter / tensor-list utilities
# ---------------------------------------------------------------------------
def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    """Return the list of parameters with ``requires_grad=True``."""
    return [p for p in model.parameters() if p.requires_grad]


def zeros_like_params(params: Sequence[Tensor]) -> List[Tensor]:
    return [torch.zeros_like(p) for p in params]


def add_(dst: List[Tensor], src: Sequence[Tensor], alpha: float = 1.0) -> None:
    """In-place ``dst += alpha * src`` for matched tensor lists."""
    for d, s in zip(dst, src):
        d.add_(s, alpha=alpha)


def scale_(tensors: List[Tensor], alpha: float) -> None:
    for t in tensors:
        t.mul_(alpha)


def flat_dot(a: Sequence[Tensor], b: Sequence[Tensor]) -> Tensor:
    """Inner product ``<a, b>`` over a list of parameter-shaped tensors."""
    return sum((ai.flatten() @ bi.flatten() for ai, bi in zip(a, b)),
               start=torch.tensor(0.0, device=a[0].device, dtype=a[0].dtype))


def flat_norm(tensors: Sequence[Tensor]) -> Tensor:
    return torch.sqrt(sum((t.pow(2).sum() for t in tensors),
                          start=torch.tensor(0.0, device=tensors[0].device,
                                             dtype=tensors[0].dtype)))


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------
def compute_gradient(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], Tensor],
    params: Sequence[nn.Parameter] | None = None,
    create_graph: bool = False,
) -> List[Tensor]:
    """Compute ``∇_θ loss_fn(model)`` and return a list of tensors.

    ``loss_fn`` is a thin closure ``model -> scalar`` so that callers can
    inject any combination of (sample, batch, validation set) without us
    knowing the data layout.
    """
    if params is None:
        params = trainable_params(model)

    model.zero_grad(set_to_none=True)
    loss = loss_fn(model)
    grads = torch.autograd.grad(loss, params, create_graph=create_graph,
                                retain_graph=create_graph, allow_unused=True)
    out: List[Tensor] = []
    for g, p in zip(grads, params):
        out.append(torch.zeros_like(p) if g is None else g.detach())
    return out


# ---------------------------------------------------------------------------
# Parameter perturbation (used by the HVP finite-difference)
# ---------------------------------------------------------------------------
@contextmanager
def perturb_parameters(
    params: Sequence[nn.Parameter],
    direction: Sequence[Tensor],
    epsilon: float,
) -> Iterator[None]:
    """Temporarily set ``p ← p + ε · direction[i]`` and restore on exit."""
    backups = [p.data.clone() for p in params]
    try:
        with torch.no_grad():
            for p, d in zip(params, direction):
                p.data.add_(d, alpha=epsilon)
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(params, backups):
                p.data.copy_(b)


# ---------------------------------------------------------------------------
# Hessian-vector product via finite differences (Eq.(6) of the paper)
# ---------------------------------------------------------------------------
def hvp_finite_diff(
    model: nn.Module,
    loss_fn: Callable[[nn.Module], Tensor],
    vector: Sequence[Tensor],
    *,
    base_grad: Sequence[Tensor] | None = None,
    epsilon: float = 1e-3,
    params: Sequence[nn.Parameter] | None = None,
) -> List[Tensor]:
    """Approximate ``H v`` where ``H = ∇² loss_fn(model)``.

    Uses the forward-difference scheme

        H v  ≈  ( ∇ loss_fn(θ + ε v) − ∇ loss_fn(θ) ) / ε

    which requires only two first-order gradient computations.  Re-uses
    ``base_grad`` (``∇ loss_fn(θ)``) if supplied to save one backward pass.
    """
    if params is None:
        params = trainable_params(model)

    if base_grad is None:
        base_grad = compute_gradient(model, loss_fn, params=params)

    # Rescale the perturbation so that its norm is comparable to ε.  This
    # mitigates numerical issues when ``vector`` has very small or very large
    # magnitude, while keeping the mathematical identity intact (we divide it
    # out below).
    v_norm = float(flat_norm(vector).item())
    if v_norm < 1e-12:
        return zeros_like_params(params)
    scale = epsilon / v_norm
    scaled_vec = [v * scale for v in vector]

    with perturb_parameters(params, scaled_vec, epsilon=1.0):
        perturbed_grad = compute_gradient(model, loss_fn, params=params)

    hv: List[Tensor] = []
    for g_pert, g_base in zip(perturbed_grad, base_grad):
        hv.append((g_pert - g_base) / scale)
    return hv


# ---------------------------------------------------------------------------
# Convenience: build a loss closure for a (data, target) pair on a model
# ---------------------------------------------------------------------------
def make_loss_closure(
    inputs: Tensor,
    targets: Tensor,
    criterion: Callable[[Tensor, Tensor], Tensor],
) -> Callable[[nn.Module], Tensor]:
    """Return ``lambda model: criterion(model(inputs), targets)``."""

    def _closure(model: nn.Module) -> Tensor:
        return criterion(model(inputs), targets)

    return _closure


def mean_loss_closure_over_loader(
    loader: Iterable,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
) -> Callable[[nn.Module], Tensor]:
    """Closure that averages the loss over every batch from ``loader``.

    Useful when we want to obtain ``∇L(V, θ)`` over the full validation set
    in one shot for moderate-sized validation splits.
    """

    cached = [(x.to(device), y.to(device)) for x, y in loader]

    def _closure(model: nn.Module) -> Tensor:
        total = torch.zeros((), device=device)
        n = 0
        for x, y in cached:
            total = total + criterion(model(x), y) * x.size(0)
            n += x.size(0)
        return total / max(n, 1)

    return _closure
