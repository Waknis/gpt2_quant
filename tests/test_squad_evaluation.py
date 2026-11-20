import json

import pytest
import torch

from src.data.squad_dataset import encode_squad_example
from src.training import train_cyclic, train_switchable
from src.training.eval_squad import em_max, evaluate, f1_max, write_results


class CharacterTokenizer:
    """Use characters as token IDs so prompt boundaries are directly visible."""
    eos_token_id = 0

    def __call__(self, text, add_special_tokens):
        return {"input_ids": list(map(ord, text))}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(token) for token in ids if token != self.eos_token_id)


@pytest.mark.parametrize("trainer", [train_cyclic, train_switchable])
def test_long_training_prompt_retains_question_and_masks_only_prompt_and_padding(trainer):
    tok = CharacterTokenizer()
    long = encode_squad_example(tok, "x" * 200, "Where?", 64, "Berlin")
    short = encode_squad_example(tok, "y", "When?", 64, "Now")
    batch = trainer.make_collate(tok)([long, short])
    assert len(long["prompt_ids"] + long["answer_ids"]) == 64
    assert tok.decode(long["prompt_ids"]).endswith("\n\nQuestion:\nWhere?\n\nAnswer: ")
    for index, sample in enumerate((long, short)):
        prompt_len = len(sample["prompt_ids"])
        end = prompt_len + len(sample["answer_ids"])
        assert batch["labels"][index, :prompt_len].eq(-100).all()
        assert batch["labels"][index, prompt_len:end].tolist() == sample["answer_ids"]
        assert batch["labels"][index, end:].eq(-100).all()
        assert batch["attention_mask"][index, :end].all()
        assert not batch["attention_mask"][index, end:].any()


def test_question_must_fit_the_prompt_budget():
    with pytest.raises(ValueError, match="cannot fit the question"):
        encode_squad_example(CharacterTokenizer(), "x", "far too long" * 10, 32)


def test_grading_uses_reference_aliases_unicode_and_token_multiplicity():
    assert em_max(" The Montréal! ", ["Quebec", "Montréal"]) == 1
    assert em_max("Montral", ["Montréal"]) == 0
    assert f1_max("red red red blue", ["green", "red blue blue"]) == pytest.approx(4 / 7)
    assert em_max(" ", ["the"]) == f1_max("", ["the"]) == 0


class PresetGenerator:
    def __init__(self):
        self.inner = self
        self.prompts = []

    def set_preset(self, preset):
        self.preset = preset

    def generate(self, input_ids, **kwargs):
        self.prompts.append(input_ids.tolist()[0])
        text = {"A": "", "B": "  Montréal! ", "C": "Paris"}[self.preset]
        new_ids = torch.tensor([list(map(ord, text)) + [0]], dtype=input_ids.dtype)
        return torch.cat([input_ids, new_ids], dim=1)


def test_paired_evaluation_grades_only_generated_tokens_and_writes_auditable_outputs(tmp_path):
    tok = CharacterTokenizer()
    model = PresetGenerator()
    rows = evaluate(model, tok, [
        {"id": "q1", "context": "Montréal " * 40, "question": "Where?", "answers": {"text": ["MTL", "Montréal"]}},
        {"id": "q2", "context": "Paris", "question": "Where?", "answers": {"text": ["Paris"]}},
    ], ["A", "B", "C"], "cpu", max_len=96)
    assert [row["id"] for row in rows] == ["q1"] * 3 + ["q2"] * 3
    assert model.prompts[0] == model.prompts[1] == model.prompts[2]
    assert tok.decode(model.prompts[0]).endswith("Question:\nWhere?\n\nAnswer: ")
    assert rows[0]["pred"] == "" and rows[0]["f1"] == rows[0]["em"] == 0
    assert rows[1]["pred_raw"] == "  Montréal! " and rows[1]["em"] == 1
    assert rows[1]["references"] == ["MTL", "Montréal"]
    assert rows[2]["pred"] == "Paris" and rows[2]["em"] == 0

    path = tmp_path / "predictions"
    summary = write_results(rows, path)
    assert json.loads(path.read_text(encoding="utf-8")) == rows
    assert json.loads((tmp_path / "predictions.summary.json").read_text()) == summary
    assert [summary[p]["count"] for p in "ABC"] == [2, 2, 2]
    assert [summary[p]["em"] for p in "ABC"] == [0, .5, .5]
    assert [summary[p]["f1"] for p in "ABC"] == [0, .5, .5]
