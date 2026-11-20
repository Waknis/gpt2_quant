"""Generate answers under each precision preset and grade them against SQuAD v1 references."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import string

import torch
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from src.data.squad_dataset import encode_squad_example
from src.models.gpt2_switchable import SwitchableGPT2


MAX_NEW_TOKENS = 16
PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    text = text.lower().translate(PUNCTUATION)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def em_max(pred: str, truths: list[str]) -> int:
    normalized = normalize_answer(pred)
    return int(bool(pred.strip()) and any(normalized == normalize_answer(t) for t in truths))


def f1_max(pred: str, truths: list[str]) -> float:
    tokens = normalize_answer(pred).split()
    counts = Counter(tokens)
    best = 0.0
    for truth in truths:
        reference = normalize_answer(truth).split()
        overlap = sum((counts & Counter(reference)).values())
        if overlap:
            best = max(best, 2 * overlap / (len(tokens) + len(reference)))
    return best


@torch.inference_mode()
def evaluate(model, tokenizer, dataset, presets, device, max_len):
    rows = []
    for idx, example in enumerate(dataset):
        encoded = encode_squad_example(
            tokenizer, example["context"], example["question"], max_len - MAX_NEW_TOKENS,
        )
        ids = torch.tensor([encoded["prompt_ids"]], device=device)
        mask = torch.ones_like(ids)
        for preset in presets:
            model.set_preset(preset)
            output = model.inner.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3,
                repetition_penalty=1.02, num_beams=4, length_penalty=0.0,
            )
            raw = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
            pred = raw.strip()
            truths = example["answers"]["text"]
            rows.append({
                "idx": idx, "id": example["id"], "preset": preset,
                "truth": truths[0] if truths else "", "references": truths,
                "pred_raw": raw, "pred": pred, "em": em_max(pred, truths), "f1": f1_max(pred, truths),
            })
    return rows


def write_results(rows, path):
    summary = {}
    for row in rows:
        stats = summary.setdefault(row["preset"], {"em": 0, "f1": 0.0, "count": 0})
        stats["em"] += row["em"]
        stats["f1"] += row["f1"]
        stats["count"] += 1
    for stats in summary.values():
        stats["em"] /= stats["count"]
        stats["f1"] /= stats["count"]
        stats["scoring"] = "squad-v1.1-generated-answer-v1"

    path.parent.mkdir(parents=True, exist_ok=True)
    # Appending to the stem also keeps the files distinct for paths without a suffix.
    summary_path = path.with_name(path.stem + ".summary.json")
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", help="Evaluate only this preset (default: all)")
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512, help="Token budget including the generated answer")
    ap.add_argument("--out", type=Path, default=Path("results/step4_eval.json"))
    ap.add_argument("--presets", default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank used when training the checkpoint")
    args = ap.parse_args()
    presets = json.loads(args.presets)
    if not presets or (args.preset is not None and args.preset not in presets):
        ap.error("select a preset defined in --presets")
    if args.max_items <= 0 or args.max_len <= MAX_NEW_TOKENS:
        ap.error(f"--max_items must be positive and --max_len must exceed {MAX_NEW_TOKENS}")
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = SwitchableGPT2("gpt2", presets=presets, lora_rank=args.rank)
    if args.max_len > model.inner.config.n_positions:
        ap.error(f"--max_len exceeds the model's context window ({model.inner.config.n_positions})")
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    dataset = load_dataset("rajpurkar/squad", split="validation")
    dataset = dataset.select(range(min(args.max_items, len(dataset))))
    selected = [args.preset] if args.preset is not None else list(presets)
    rows = evaluate(model, tokenizer, dataset, selected, device, args.max_len)
    summary = write_results(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out} | summary: {summary}")


if __name__ == "__main__":
    main()
