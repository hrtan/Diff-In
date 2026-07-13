"""High-level *Diff-In* influence estimator.

Implements the checkpoint-accelerated formulation from Eq.(9) of the paper:

    I_θ(z)   = Σ_{t∈T_m}  a_t · ( H^t_{B_t} G^t_z  +  H^t_z G^t_{B_t} )
    I(z, V)  = Σ_{t∈T_m}  a_t · ⟨ ∇L(V, θ_t),  H^t_{B_t} G^t_z + H^t_z G^t_{B_t} ⟩

with coefficient

    a_t = − t · η_t² / (N · m)

and Hessian-vector products approximated by finite differences (Eq.(6)).

The estimator is purposefully *checkpoint-driven*: callers supply a list of
``CheckpointSpec`` objects describing each saved training state.  This keeps
the algorithm decoupled from the training script, so the same estimator can
be reused with checkpoints produced by any pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from . import utils


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CheckpointSpec:
    """Information needed to evaluate Diff-In at one training step.

    Attributes
    ----------
    path        Path to a ``state_dict`` saved during training.
    step        The training iteration ``t`` at which the checkpoint was taken.
    lr          The learning rate ``η_t`` in effect at that iteration.
    """

    path: str
    step: int
    lr: float


@dataclass
class InfluenceResult:
    """Container returned by :meth:`DiffInEstimator.compute`.

    ``influence_params`` matches the structure of ``model.parameters()`` and
    represents ``I_θ(z)``; ``influence_loss`` is the scalar ``I(z, V)``
    (``None`` if no validation closure was provided).
    """

    influence_loss: Optional[float] = None
    influence_params: Optional[List[Tensor]] = None
    per_checkpoint_loss: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The estimator
# ---------------------------------------------------------------------------
class DiffInEstimator:
    """Differential-Influence (Diff-In) estimator.

    Parameters
    ----------
    model_factory
        Callable returning a *fresh* ``nn.Module`` instance whose architecture
        matches the saved checkpoints.  A fresh model is built once and reused
        across checkpoints (its weights are reloaded each time).
    checkpoints
        Ordered list of :class:`CheckpointSpec` describing where to find each
        saved model state, what step it corresponds to, and which learning
        rate produced it.
    train_loader
        DataLoader sampling random training batches; used as a stochastic
        proxy for ``∇L(D, θ_t)`` (the paper does the same — see Sec. 6.1).
    dataset_size
        ``N``, the total number of training samples.  Appears in the Diff-In
        coefficient ``a_t = −t · η_t² / (N · m)``.
    criterion
        Per-sample / per-batch loss, e.g. ``nn.CrossEntropyLoss()`` (reduction
        must be ``"mean"`` for batches).
    device
        Where the model and tensors live.
    epsilon
        Finite-difference step size used inside ``Hv`` approximation.
    batches_per_checkpoint
        How many random batches we average to approximate ``G^t_{B_t}`` (and
        the associated ``H^t_{B_t}``) at each checkpoint.  A small value (1-4)
        works well in practice.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        checkpoints: Sequence[CheckpointSpec],
        train_loader: DataLoader,
        dataset_size: int,
        criterion: Callable[[Tensor, Tensor], Tensor],
        device: torch.device | str = "cpu",
        epsilon: float = 1e-3,
        batches_per_checkpoint: int = 1,
    ) -> None:
        if len(checkpoints) == 0:
            raise ValueError("At least one checkpoint is required.")
        self.model_factory = model_factory
        self.checkpoints = list(checkpoints)
        self.train_loader = train_loader
        self.dataset_size = int(dataset_size)
        self.criterion = criterion
        self.device = torch.device(device)
        self.epsilon = float(epsilon)
        self.batches_per_checkpoint = int(batches_per_checkpoint)

        self._model = self.model_factory().to(self.device)
        self._params = utils.trainable_params(self._model)

    # ------------------------------------------------------------------ utils
    def _load_checkpoint(self, ckpt: CheckpointSpec) -> None:
        state = torch.load(ckpt.path, map_location=self.device, weights_only=True)
        # Allow either a raw state-dict or a dict with a ``model`` field.
        if isinstance(state, dict) and "model" in state and "state_dict" not in state:
            state = state["model"]
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state)
        self._params = utils.trainable_params(self._model)

    def _sample_batches(self) -> List[Tuple[Tensor, Tensor]]:
        """Draw ``batches_per_checkpoint`` random batches from ``train_loader``."""
        batches: List[Tuple[Tensor, Tensor]] = []
        it = iter(self.train_loader)
        for _ in range(self.batches_per_checkpoint):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(self.train_loader)
                x, y = next(it)
            batches.append((x.to(self.device), y.to(self.device)))
        return batches

    # ------------------------------------------------------- core per-step term
    def _difference_term(
        self,
        ckpt: CheckpointSpec,
        z_inputs: Tensor,
        z_targets: Tensor,
    ) -> List[Tensor]:
        """Compute the bracketed quantity ``H^t_{B_t} G^t_z + H^t_z G^t_{B_t}``.

        Returns a list of tensors with the same shape as ``model.parameters()``.
        """
        self._load_checkpoint(ckpt)
        params = self._params

        # ---- gradient on the single training sample z -----------------------
        loss_z = utils.make_loss_closure(z_inputs, z_targets, self.criterion)
        g_z = utils.compute_gradient(self._model, loss_z, params=params)

        # ---- accumulate over (random) batches as proxy for ∇L(D, θ_t) -------
        bracket = utils.zeros_like_params(params)
        batches = self._sample_batches()
        for bx, by in batches:
            loss_b = utils.make_loss_closure(bx, by, self.criterion)
            g_b = utils.compute_gradient(self._model, loss_b, params=params)

            #   H^t_{B_t} · G^t_z       (sensitivity of batch grad to z's grad)
            hbt_gz = utils.hvp_finite_diff(
                self._model, loss_b, g_z,
                base_grad=g_b, epsilon=self.epsilon, params=params,
            )
            #   H^t_z · G^t_{B_t}       (sensitivity of z's grad to batch grad)
            hz_gb = utils.hvp_finite_diff(
                self._model, loss_z, g_b,
                base_grad=g_z, epsilon=self.epsilon, params=params,
            )
            utils.add_(bracket, hbt_gz, alpha=1.0)
            utils.add_(bracket, hz_gb, alpha=1.0)

        # average over the proxy batches so the scale matches ∇L(D, ·)
        if len(batches) > 1:
            utils.scale_(bracket, 1.0 / len(batches))
        return bracket

    # ------------------------------------------------------------ public API
    def compute(
        self,
        z_inputs: Tensor,
        z_targets: Tensor,
        validation_grad_fn: Optional[Callable[[nn.Module], List[Tensor]]] = None,
        want_params_influence: bool = True,
    ) -> InfluenceResult:
        """Compute Diff-In influence for a single sample ``z``.

        Parameters
        ----------
        z_inputs, z_targets
            Inputs / labels of the single training sample under study.
            ``z_inputs`` should have a leading batch dim of size 1.
        validation_grad_fn
            Callable returning ``∇L(V, θ)`` given the (already-loaded) model.
            Provide e.g. ``lambda m: utils.compute_gradient(m, val_closure)``
            to obtain ``I(z, V)``.  If ``None``, only ``I_θ(z)`` is computed.
        want_params_influence
            If ``False``, skip accumulating the (potentially large) tensor
            list ``I_θ(z)``.
        """
        z_inputs = z_inputs.to(self.device)
        z_targets = z_targets.to(self.device)

        N = self.dataset_size
        m = len(self.checkpoints)

        result = InfluenceResult()
        if want_params_influence:
            result.influence_params = utils.zeros_like_params(self._params)
        accum_loss = 0.0

        for ckpt in self.checkpoints:
            bracket = self._difference_term(ckpt, z_inputs, z_targets)

            # Diff-In coefficient (Eq.(9))
            a_t = -ckpt.step * (ckpt.lr ** 2) / (N * m)

            if validation_grad_fn is not None:
                val_grad = validation_grad_fn(self._model)
                contrib = float(a_t * utils.flat_dot(val_grad, bracket).item())
                accum_loss += contrib
                result.per_checkpoint_loss.append(contrib)

            if want_params_influence:
                utils.add_(result.influence_params, bracket, alpha=a_t)

        if validation_grad_fn is not None:
            result.influence_loss = accum_loss
        return result

    # ---------------------------------------------- convenience entry points
    def influence_on_loss(
        self,
        z_inputs: Tensor,
        z_targets: Tensor,
        validation_loader: DataLoader,
    ) -> float:
        """Return scalar ``I(z, V)`` over the validation loader."""
        val_closure = utils.mean_loss_closure_over_loader(
            validation_loader, self.criterion, self.device
        )

        def _grad_fn(model: nn.Module) -> List[Tensor]:
            return utils.compute_gradient(model, val_closure, params=self._params)

        out = self.compute(z_inputs, z_targets,
                           validation_grad_fn=_grad_fn,
                           want_params_influence=False)
        assert out.influence_loss is not None
        return out.influence_loss

    def influence_on_params(
        self,
        z_inputs: Tensor,
        z_targets: Tensor,
    ) -> List[Tensor]:
        """Return ``I_θ(z)`` as a list of parameter-shaped tensors."""
        out = self.compute(z_inputs, z_targets,
                           validation_grad_fn=None,
                           want_params_influence=True)
        assert out.influence_params is not None
        return out.influence_params

    def self_influence(self, z_inputs: Tensor, z_targets: Tensor) -> float:
        """Influence of ``z`` on its own loss (paper Eq.(12))."""

        def _grad_fn(model: nn.Module) -> List[Tensor]:
            closure = utils.make_loss_closure(z_inputs.to(self.device),
                                              z_targets.to(self.device),
                                              self.criterion)
            return utils.compute_gradient(model, closure, params=self._params)

        out = self.compute(z_inputs, z_targets,
                           validation_grad_fn=_grad_fn,
                           want_params_influence=False)
        assert out.influence_loss is not None
        return out.influence_loss
