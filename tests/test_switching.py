
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.lora.lora_linear import LoRALinear
from src.models.gpt2_switchable import SwitchableGPT2


@pytest.fixture(scope="module")
def pretrained_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("tiny-gpt2")
    config = GPT2Config(
        vocab_size=32, n_positions=16, n_embd=8, n_layer=1, n_head=2,
        bos_token_id=0, eos_token_id=1,
    )
    GPT2LMHeadModel(config).save_pretrained(path)
    return path


@pytest.fixture
def model(pretrained_dir):
    presets = {
        "A": {"w_bits": 8, "a_bits": 8},
        "B": {
            "w_bits": 4, "a_bits": 4,
            "layers": [{"pattern": r"c_attn$", "w_bits": 3}],
        },
    }
    return SwitchableGPT2(pretrained_dir, presets=presets)


def test_switching(model):
    for preset in ("B", "A"):
        model.set_preset(preset)
        assert model.active_preset == preset
        assert model.inner.lm_head.weight is model.inner.transformer.wte.weight
        projections = {
            name: layer for name, layer in model.inner.named_modules() if isinstance(layer, LoRALinear)
        }
        assert len(projections) == 4
        for name, layer in projections.items():
            default_bits = model.presets[preset]["a_bits"]
            expected_w = 3 if preset == "B" and name.endswith("c_attn") else default_bits
            assert (layer.active, layer.base.w_bits, layer.base.a_bits) == (preset, expected_w, default_bits)
        with torch.no_grad():
            logits = model(torch.tensor([[2, 3]])).logits
        assert logits.shape == (1, 2, 32)
        assert torch.isfinite(logits).all()


def test_random_preset_uses_supplied_generator(model):
    rng = torch.Generator().manual_seed(123)
    global_state = torch.get_rng_state()
    rng_state = rng.get_state()
    model.random_preset(rng=rng)
    assert not torch.equal(rng.get_state(), rng_state)
    assert torch.equal(torch.get_rng_state(), global_state)


@pytest.mark.parametrize("legacy", [False, True])
def test_checkpoint_round_trip(model, pretrained_dir, legacy):
    with torch.no_grad():
        model.inner.transformer.h[0].attn.c_attn.adapters["B"]["B"].fill_(0.75)
    checkpoint = model.state_dict()
    if legacy:
        checkpoint["inner.lm_head.base.fc.weight"] = checkpoint.pop("inner.lm_head.weight")

    restored = SwitchableGPT2(pretrained_dir, presets=model.presets)
    restored.load_state_dict(checkpoint)
    assert restored.inner.lm_head.weight is restored.inner.transformer.wte.weight
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[name])


def test_incomplete_checkpoint_reports_keys(model):
    checkpoint = model.state_dict()
    missing = "inner.transformer.h.0.attn.c_attn.adapters.A.B"
    checkpoint["unused.weight"] = checkpoint.pop(missing)
    with pytest.raises(RuntimeError, match="Missing key"):
        model.load_state_dict(checkpoint)
    result = model.load_state_dict(checkpoint, strict=False)
    assert result.missing_keys == [missing]
    assert result.unexpected_keys == ["unused.weight"]
