"""
Step 2: Low-Rank Adaptation (LoRA) with Multiple Adapters
Paper Reference: LoRA (Microsoft, 2021) - Section 4.1
- Low-rank decomposition: ΔW = BA where B ∈ R^(d×r), A ∈ R^(r×k)
- Initialization: A ~ N(0, σ²), B = 0 (so ΔW = 0 at start)
- Scaling: α/r where α is constant (set to rank by default)
- Forward: h = W₀x + BAx (base layer + LoRA adaptation)
- Multiple adapters: One per quantization preset for switchable precision
"""
import torch
import torch.nn as nn
from src.quantization.fake_quant import QuantLinear

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, w_bits=8, a_bits=8):
        super().__init__()
        self.base = QuantLinear(in_features, out_features, bias=bias, w_bits=w_bits, a_bits=a_bits)
        self.adapters = nn.ModuleDict()
        self.active = None

    def add_adapter(self, name: str, r: int = 8, alpha: int = 8, dropout: float = 0.0, init_scale: float = 1e-3):
        if name in self.adapters:
            return
        A = nn.Parameter(torch.zeros(self.base.in_features, r))
        B = nn.Parameter(torch.zeros(r, self.base.out_features))
        nn.init.normal_(A, std=init_scale)
        nn.init.zeros_(B)
        self.adapters[name] = nn.ParameterDict({
            "A": A,
            "B": B,
            "alpha": nn.Parameter(torch.tensor(float(alpha)), requires_grad=False),
            "dropout": nn.Parameter(torch.tensor(float(dropout)), requires_grad=False),
        })

    def set_active(self, name: str | None):
        self.active = name

    def set_bits(self, w_bits=None, a_bits=None):
        self.base.set_bits(w_bits=w_bits, a_bits=a_bits)

    def forward(self, x):
        y = self.base(x)
        if self.active is None or self.active not in self.adapters:
            return y
        pack = self.adapters[self.active]
        A, B = pack["A"], pack["B"]
        alpha = float(pack["alpha"].detach().item())
        r = A.shape[1]
        return y + (x @ A @ B) * (alpha / max(1, r))