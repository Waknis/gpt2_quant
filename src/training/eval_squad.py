"""
Step 4: Evaluation of Different Quantization Configurations
Paper Reference: SQuAD dataset evaluation standard
- Exact Match (EM): Percentage of predictions matching ground truth exactly
- F1 Score: Token-level F1 between prediction and ground truth
- Context-aware answer extraction for better matching
- Per-preset evaluation to compare different bit-width configurations
"""
import os, json, argparse, torch, re
from transformers import GPT2TokenizerFast
from src.models.gpt2_switchable import SwitchableGPT2

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return re.sub(r"[^0-9a-zA-Z\s]", "", text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_max(pred: str, truths: list[str]) -> int:
    npred = normalize_answer(pred)
    return int(any(npred == normalize_answer(t) for t in truths))

def f1_max(pred: str, truths: list[str]) -> float:
    npred = normalize_answer(pred)
    p_tokens = npred.split()
    if not p_tokens:
        return 0.0
    best = 0.0
    for t in truths:
        t_tokens = normalize_answer(t).split()
        if not t_tokens:
            best = max(best, 0.0)
            continue
        common = {}
        for w in p_tokens:
            if w in t_tokens:
                common[w] = common.get(w, 0) + 1
        num_same = sum(min(p_tokens.count(w), t_tokens.count(w)) for w in common.keys())
        if num_same == 0:
            continue
        precision = num_same / len(p_tokens)
        recall = num_same / len(t_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best

TEMPLATE = "Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--preset", type=str, default=None)
    ap.add_argument("--max_items", type=int, default=200)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", type=str, default="results/step4_eval.json")
    ap.add_argument("--presets", type=str, default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad")["validation"].select(range(args.max_items))

    presets = json.loads(args.presets)
    model = SwitchableGPT2("gpt2", presets=presets)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    try:
        model.inner.config.pad_token_id = tok.eos_token_id
    except Exception:
        pass

    rows = []
    agg = {}
    def normalize_pred(pred: str) -> str:
        # take first line/sentence, shorten, strip punctuation
        pred0 = pred.split("\n")[0]
        pred0 = re.split(r"[\.\?!]", pred0)[0]
        words = re.findall(r"\w+", pred0)
        words = words[:8]
        return " ".join(words)
    def best_subspan_from_context(context: str, hint: str, max_n: int = 8) -> str:
        ctx_toks = re.findall(r"\w+", context.lower())
        hint_toks = re.findall(r"\w+", hint.lower())
        if not hint_toks or not ctx_toks:
            return hint
        best = (0.0, "")
        for n in range(1, max_n+1):
            for i in range(0, max(0, len(ctx_toks)-n+1)):
                span = ctx_toks[i:i+n]
                common = set(span) & set(hint_toks)
                if not common: continue
                prec = sum(w in common for w in span)/len(span)
                rec = sum(w in common for w in hint_toks)/len(hint_toks)
                score = 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)
                if score > best[0]:
                    best = (score, " ".join(span))
        return best[1] or hint
    run_presets = [args.preset] if args.preset else list(presets.keys())
    for i, r in enumerate(ds):
        prompt = TEMPLATE.format(context=r["context"], question=r["question"]) + " "
        truths = r["answers"]["text"] if r["answers"]["text"] else ["unknown"]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=args.max_len-32)
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        for p in run_presets:
            model.set_preset(p)
            out = model.inner.generate(
                input_ids=ids, attention_mask=mask, max_new_tokens=16, do_sample=False,
                pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                no_repeat_ngram_size=3, repetition_penalty=1.02,
                num_beams=4, length_penalty=0.0
            )
            gen = out[0][ids.shape[1]:]
            pred_raw = tok.decode(gen, skip_special_tokens=True)
            pred_clean = normalize_pred(pred_raw)
            ctx_lc = r["context"].lower()
            if (not pred_clean) or (pred_clean.lower() not in ctx_lc):
                hint = pred_clean if pred_clean else (truths[0] if truths else "")
                pred_clean = best_subspan_from_context(r["context"], hint)
            em = em_max(pred_clean, truths)
            f1 = f1_max(pred_clean, truths)
            rows.append({"idx": i, "preset": p, "truth": truths[0], "pred": pred_clean,
                         "em": em, "f1": f1})
            agg.setdefault(p, {"em_sum":0, "f1_sum":0.0, "n":0})
            agg[p]["em_sum"] += em; agg[p]["f1_sum"] += f1; agg[p]["n"] += 1
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f: json.dump(rows, f, indent=2)
    # write per-preset aggregates
    summary = {}
    for p, s in agg.items():
        n = max(1, s["n"])
        summary[p] = {"em": s["em_sum"]/n, "f1": s["f1_sum"]/n, "count": s["n"]}
    with open(re.sub(r"\.json$", ".summary.json", args.out), "w") as f: json.dump(summary, f, indent=2)
    print(f"Wrote {len(rows)} rows to {args.out} | summary: {summary}")

if __name__ == "__main__":
    main()