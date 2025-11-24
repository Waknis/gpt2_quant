
import torch
from src.lora.lora_linear import LoRALinear

def test_lora_routing():
    ll = LoRALinear(16, 8)
    ll.add_adapter("A", r=4, alpha=4); ll.add_adapter("B", r=4, alpha=4)
    x = torch.randn(2,16); ll.set_active("A"); y1 = ll(x)
    ll.set_active("B"); y2 = ll(x)
    assert (y1-y2).abs().sum() > 0
