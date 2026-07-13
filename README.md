# Diff-In: Differential Influence Estimator

This repository implements **Diff-In**, the data-influence estimator proposed
in *"Understanding Data Influence with Differential Approximation"*
(Tan et al., arXiv:2508.14648). Diff-In approximates the influence of a
single training sample by accumulating the **differences in influence between
consecutive learning steps**, yielding a *second-order* estimator whose cost
matches that of first-order methods.

The checkpoint-accelerated estimator (paper Eq.(9)) is:

```
I_θ(z)   = Σ_{t∈T_m}  -t·η_t² / (N·m) · ( H^t_{B_t} G^t_z + H^t_z G^t_{B_t} )
I(z, V)  = Σ_{t∈T_m}  -t·η_t² / (N·m) · ⟨ ∇L(V, θ_t),  H^t_{B_t} G^t_z + H^t_z G^t_{B_t} ⟩
```

Each Hessian–vector product `H v` is approximated by a first-order finite
difference of gradients (paper Eq.(6)):

```
H v ≈ ( ∇L(θ + ε v) − ∇L(θ) ) / ε
```

so the whole pipeline only ever needs ordinary back-propagation — no explicit
Hessian, no inverse-Hessian solve, no convexity assumption.

---

## Project layout

```
DiffIn/
├── diff_in/                       # Core algorithm package
│   ├── __init__.py
│   ├── utils.py                   # Gradient / HVP / parameter-perturbation helpers
│   └── estimator.py               # DiffInEstimator (high-level API)
├── train_cifar.py                 # CIFAR-10 training + evenly-spaced checkpoint dumps
├── compute_influence_example.py   # Influence computation for one training sample
├── requirements.txt
└── README.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Train ResNet-18 on CIFAR-10 (saves m=5 evenly-spaced checkpoints).
python train_cifar.py --epochs 10 --num-checkpoints 5 --batch-size 128

# 2) Estimate Diff-In influence for training sample #42
#    (self-influence + influence on validation loss).
python compute_influence_example.py --sample-index 42
```

## API in a nutshell

```python
from diff_in import DiffInEstimator, CheckpointSpec

estimator = DiffInEstimator(
    model_factory=build_resnet18,
    checkpoints=[CheckpointSpec(path=..., step=..., lr=...), ...],
    train_loader=train_loader,
    dataset_size=len(train_set),
    criterion=nn.CrossEntropyLoss(),
    device="cuda",
    epsilon=1e-3,             # finite-difference step
    batches_per_checkpoint=1, # proxy batches used for ∇L(D, θ_t)
)

i_loss   = estimator.influence_on_loss(z_x, z_y, val_loader)   # I(z, V)
i_params = estimator.influence_on_params(z_x, z_y)             # I_θ(z)
i_self   = estimator.self_influence(z_x, z_y)                  # I(z, z)
```

## Citation

If you find this work useful, please cite the paper:

```bibtex
@article{tan2026diffin,
  title   = {Understanding Data Influence with Differential Approximation},
  author  = {Tan, Haoru and Wu, Sitong and Wu, Xiuzhe and Wang, Wang and
             Zhao, Bo and Xie, Zeke and Xia, Gui-Song and Qi, Xiaojuan},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year    = {2026}
}
```
