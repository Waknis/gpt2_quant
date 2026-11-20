import re
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from transformers.pytorch_utils import Conv1D
from src.lora.lora_linear import LoRALinear

def replace_linear_with_lora(module: nn.Module, presets: dict, lora_rank: int, lora_alpha: int):
    """
    Replace both nn.Linear and Conv1D layers with LoRA-enabled versions.

    Note: GPT-2 uses Conv1D layers (transposed linear layers) in attention and MLP blocks.
    Conv1D weight shape: (in_features, out_features) vs nn.Linear: (out_features, in_features)
    """
    for name, child in list(module.named_children()):
        # Check for both nn.Linear and Conv1D (GPT-2 uses Conv1D)
        if isinstance(child, nn.Linear):
            # Preserve GPT-2 output embedding weight tying: do not wrap lm_head
            if name == "lm_head":
                continue
            new = LoRALinear(child.in_features, child.out_features, bias=(child.bias is not None), w_bits=8, a_bits=8)
            with torch.no_grad():
                new.base.fc.weight.copy_(child.weight)
                if child.bias is not None:
                    new.base.fc.bias.copy_(child.bias)
            for k in presets.keys():
                new.add_adapter(k, r=lora_rank, alpha=lora_alpha, dropout=0.0)
            setattr(module, name, new)
        elif isinstance(child, Conv1D):
            # Conv1D: nf=out_features, nx=in_features, weight shape is (nx, nf) = (in, out)
            in_features, out_features = child.weight.shape
            new = LoRALinear(in_features, out_features, bias=True, w_bits=8, a_bits=8)
            with torch.no_grad():
                # Conv1D weight is (in, out), nn.Linear weight is (out, in), so transpose
                new.base.fc.weight.copy_(child.weight.t())
                new.base.fc.bias.copy_(child.bias)
            for k in presets.keys():
                new.add_adapter(k, r=lora_rank, alpha=lora_alpha, dropout=0.0)
            setattr(module, name, new)
        else:
            replace_linear_with_lora(child, presets, lora_rank, lora_alpha)

class SwitchableGPT2(nn.Module):
    def __init__(self, model_name="gpt2", presets: dict | None = None, lora_rank: int = 8, lora_alpha: int = 8):
        super().__init__()
        self.inner = GPT2LMHeadModel.from_pretrained(model_name)
        self.presets = presets or {"fp16": {"w_bits": 16, "a_bits": 16}, "int8": {"w_bits": 8, "a_bits": 8}}
        self.active_preset = list(self.presets.keys())[0]
        replace_linear_with_lora(self.inner, self.presets, lora_rank, lora_alpha)
        self._apply_preset_bits(self.active_preset)
        self._route_adapters(self.active_preset)

    @torch.no_grad()
    def _apply_preset_bits(self, preset_name: str):
        cfg = self.presets[preset_name]
        rules = cfg.get("layers") if isinstance(cfg, dict) else None
        default_w = int(cfg.get("w_bits", 8)) if isinstance(cfg, dict) else 8
        default_a = int(cfg.get("a_bits", 8)) if isinstance(cfg, dict) else 8
        compiled = [
            (re.compile(rule["pattern"]), int(rule.get("w_bits", default_w)), int(rule.get("a_bits", default_a)))
            for rule in (rules or [])
        ]
        for name, layer in self.inner.named_modules():
            if not isinstance(layer, LoRALinear):
                continue
            # Configure each projection once. Its nested quantizer shares the same bits.
            for pattern, w_bits, a_bits in compiled:
                if pattern.search(name) or pattern.search(name + ".base"):
                    layer.set_bits(w_bits=w_bits, a_bits=a_bits)
                    break
            else:
                layer.set_bits(w_bits=default_w, a_bits=default_a)

    def _route_adapters(self, preset_name: str):
        for m in self.modules():
            if hasattr(m, "set_active"):
                m.set_active(preset_name)

    def set_preset(self, preset_name: str):
        assert preset_name in self.presets, f"Unknown preset {preset_name}"
        self.active_preset = preset_name
        self._apply_preset_bits(preset_name)
        self._route_adapters(preset_name)

    def random_preset(self, rng: torch.Generator | None = None, weights: list[float] | None = None):
        keys = list(self.presets.keys())
        probs = torch.tensor(weights or [1.0/len(keys)]*len(keys))
        idx = torch.multinomial(probs, 1, generator=rng).item()
        self.set_preset(keys[idx])

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # Older checkpoints wrapped the output head. GPT-2 now ties it to the embeddings.
        legacy_key = prefix + "inner.lm_head.base.fc.weight"
        if legacy_key in state_dict:
            legacy_weight = state_dict.pop(legacy_key)
            embedding = state_dict.get(prefix + "inner.transformer.wte.weight", legacy_weight)
            state_dict.setdefault(prefix + "inner.lm_head.weight", embedding)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)
