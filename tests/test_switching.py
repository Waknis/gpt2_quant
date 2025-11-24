
from transformers import GPT2TokenizerFast
from src.models.gpt2_switchable import SwitchableGPT2

def test_switching():
    m = SwitchableGPT2("gpt2", presets={"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4}})
    m.set_preset("B")
    assert m.active_preset == "B"
