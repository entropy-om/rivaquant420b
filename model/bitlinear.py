"""BitNet b1.58 ternary linear layer.

Adapted from the reference math in kyegomez/BitNet (github.com/kyegomez/BitNet,
bitnet/bitlinear.py) and Microsoft's BitNet b1.58 paper (arXiv:2402.17764) — not
reimplemented from memory. Weights are ternary {-1, 0, 1} scaled by their
absmean; activations are per-token int8. Both use a straight-through estimator
(STE): the forward pass uses the quantized value, the backward pass treats
quantization as identity so gradients flow to the full-precision shadow
weights that actually get updated by the optimizer.
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
    """Ternary quantization: sign(w - mean(w)) * mean(|w|)."""
    scale = w.abs().mean()
    e = w.mean()
    u = (w - e).sign() * scale
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
    """Drop-in replacement for nn.Linear with BitNet b1.58 ternary weights."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        x_norm = self.norm(x)
        # STE: forward uses the quantized value, gradient passes through as identity.
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        return F.linear(x_quant, w_quant, self.bias)
