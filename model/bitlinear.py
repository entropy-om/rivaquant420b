"""BitNet b1.58 ternary linear layer.

Adapted from the reference math in kyegomez/BitNet (github.com/kyegomez/BitNet,
bitnet/bitlinear.py) and Microsoft's BitNet b1.58 paper (arXiv:2402.17764) — not
reimplemented from memory. Weights are nominally ternary {-1, 0, 1} scaled by
their absmean; activations are per-token int8. Both use a straight-through
estimator (STE): the forward pass uses the quantized value, the backward pass
treats quantization as identity so gradients flow to the full-precision shadow
weights that actually get updated by the optimizer.

Measured, not assumed: `weight_quant` below is sign()-based, and `.sign()` on
a continuous float essentially never returns exactly 0 — checked against a
real trained checkpoint (rivaquant420b 162m, 162,129,408 quantized weight
elements): 9 exact zeros, 0.0000055%. So this implementation has, in
practice, always been functionally 1-bit binary, not truly ternary — "b1.58"
in name, ~1.0 bits in measured behavior. `binary_weight_quant` below makes
that explicit and provable (no possible zero, ever) rather than an
accident of floating point; see `RivaQuantConfig.binary_weights`.
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def activation_quant(x: Tensor) -> Tensor:
    """Per-token int8 quantization, no grouping needed."""
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w: Tensor) -> Tensor:
    """Nominally-ternary quantization: sign(w - mean(w)) * mean(|w|). See the
    module docstring — in measured practice this is ~binary already, since
    .sign() on a continuous float essentially never lands on exact 0."""
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
    return u


def binary_weight_quant(w: Tensor) -> Tensor:
    """Provably 1-bit: every element is exactly +scale or -scale, no
    exceptions. Unlike weight_quant's .sign() (which *could* in principle
    return exactly 0 for w == mean(w)), torch.where's boundary case is
    deterministic — ties round to +1, not to an ambiguous third value."""
    scale = w.abs().mean()
    e = w.mean()
    u = torch.where(w >= e, torch.ones_like(w), -torch.ones_like(w)) * scale
    return u


class RMSNorm(nn.Module):
    """torch.nn.RMSNorm needs torch>=2.4; the RunPod training image ships
    2.1.0. Same math, no version dependency: x / rms(x) * weight."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class BitLinear(nn.Linear):
    """Drop-in replacement for nn.Linear with BitNet b1.58 ternary weights
    (or, with binary=True, provably 1-bit weights — see bitlinear.py's
    module docstring for why the two are nearly indistinguishable in
    practice, and RivaQuantConfig.binary_weights for which stages use which)."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, binary: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)
        self.binary = binary

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.norm(x)
        quant_fn = binary_weight_quant if self.binary else weight_quant
        # STE: forward uses the quantized value, gradient passes through as identity.
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (quant_fn(w) - w).detach()
        return F.linear(x_quant, w_quant, self.bias)
