
import torch
from src.quantization.fake_quant import quantize_activation, quantize_weight_per_channel

def test_activation_ste():
    x = torch.randn(4,4, requires_grad=True)
    y = quantize_activation(x, 4); y.sum().backward()
    assert x.grad is not None

def test_weight_per_channel():
    w = torch.randn(8,4, requires_grad=True)
    wq = quantize_weight_per_channel(w, 4, dim=0)
    (wq.sum()).backward()
    assert w.grad is not None
