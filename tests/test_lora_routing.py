
import torch

from src.lora.lora_linear import LoRALinear


def test_lora_routing():
    layer = LoRALinear(3, 2)
    for name in ("A", "B"):
        layer.add_adapter(name, r=1, alpha=2)
    x = torch.tensor([[1., 2., 3.]])
    base = layer(x)

    # Fresh adapters are zero updates. Apply a known update to B only.
    with torch.no_grad():
        layer.adapters["B"]["A"].fill_(1.0)
        layer.adapters["B"]["B"].copy_(torch.tensor([[1., -1.]]))
    layer.set_active("B")
    torch.testing.assert_close(layer(x), base + torch.tensor([[12., -12.]]))
    layer.set_active("A")
    torch.testing.assert_close(layer(x), base)
