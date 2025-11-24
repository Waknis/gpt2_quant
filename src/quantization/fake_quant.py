"""
Step 1: Per-layer Quantization with Switchable Precision
Paper Reference: QAT-LLM & Quantization-Aware Training literature
- Symmetric min-max quantization scheme
- Per-channel weight quantization for better accuracy
- Straight-Through Estimator (STE) for gradient flow
- Support for different bit-widths per layer (w_bits, a_bits)
"""
import torch
import torch.nn as nn

def symmetric_minmax_scale(x: torch.Tensor, bits: int, eps: float = 1e-8):
    max_abs = x.detach().abs().max()
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / (qmax + eps)
    return torch.clamp(scale, min=eps)

def symmetric_minmax_scale_per_channel(w: torch.Tensor, bits: int, dim: int = 0, eps: float = 1e-8):
    max_abs = w.detach().abs().amax(dim=dim, keepdim=True)
    qmax = (1 << (bits - 1)) - 1
    scale = max_abs / (qmax + eps)
    return torch.clamp(scale, min=eps)

class STEQuant(torch.autograd.Function):
    """
    Straight-Through Estimator for Quantization
    Paper: "Estimating or Propagating Gradients Through Stochastic Neurons" (Bengio et al., 2013)
    Forward: Quantize with rounding
    Backward: Pass gradient straight through (∂L/∂x_q ≈ ∂L/∂x)
    """
    @staticmethod
    def forward(ctx, x, scale):
        return torch.round(x / scale) * scale
    @staticmethod
    def backward(ctx, g):
        return g, None  # Straight-through: gradient flows unchanged

def quantize_activation(x: torch.Tensor, bits: int):
    if bits >= 16:
        return x
    s = symmetric_minmax_scale(x, bits)
    return STEQuant.apply(x, s)

def quantize_weight_per_channel(w: torch.Tensor, bits: int, dim: int = 0):
    if bits >= 16:
        return w
    s = symmetric_minmax_scale_per_channel(w, bits, dim=dim)
    return STEQuant.apply(w, s)

class QuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_bits=8, a_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)

    def set_bits(self, w_bits=None, a_bits=None):
        if w_bits is not None:
            self.w_bits = int(w_bits)
        if a_bits is not None:
            self.a_bits = int(a_bits)

    def forward(self, x):
        xq = quantize_activation(x, self.a_bits) if self.a_bits < 16 else x
        wq = quantize_weight_per_channel(self.fc.weight, self.w_bits, dim=0) if self.w_bits < 16 else self.fc.weight
        return torch.nn.functional.linear(xq, wq, self.fc.bias)