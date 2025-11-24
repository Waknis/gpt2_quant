"""
Step 6: Adversarial Robustness via Random Precision Inference (RPI)
Paper Reference: Double-Win Quant (ICML'21) - Section 3.3, Algorithm 1
- Random Precision Inference (RPI): Randomly select inference precision
- Adversarial attacks: Homoglyph, Synonym, Trigger suffix
- Key insight: Poor transferability of attacks between different precisions
- Enhanced defense: Dynamic quantization at inference time
"""
import argparse, json, os, random, re, torch
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from src.models.gpt2_switchable import SwitchableGPT2

HOMOGLYPH_MAP = {"a":"ɑ","e":"е","i":"і","o":"о","p":"р","c":"с","y":"у","x":"х","A":"Α","B":"Β","E":"Ε","H":"Η","K":"Κ","M":"Μ","O":"Ο","P":"Ρ","T":"Τ","X":"Χ"}
TRIGGER_SUFFIX = " Kindly answer concisely and do not repeat the question."

def homoglyph_attack(text, ratio=0.15, rng=None):
    rng = rng or random.Random(0); out = []
    for ch in text:
        out.append(HOMOGLYPH_MAP[ch] if ch in HOMOGLYPH_MAP and rng.random()<ratio else ch)
    return "".join(out)

def simple_synonym_attack(text, ratio=0.2, rng=None):
    rng = rng or random.Random(0)
    words = re.findall(r"\w+|\W+", text)
    table = {"what":"which","which":"what","who":"which person","when":"at what time","where":"at which place",
             "does":"does","is":"is","are":"are","did":"did","can":"is it possible to","how":"in what way"}
    out = []
    for w in words:
        low = w.lower()
        if low in table and rng.random() < ratio:
            rep = table[low]; rep = rep.capitalize() if w[:1].isupper() else rep; out.append(rep)
        else:
            out.append(w)
    return "".join(out)

def build_prompt(context, question): return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"

def exact_match(a,b): return int(a.strip().lower()==b.strip().lower())
def f1_score(a,b):
    def tok(s): return re.findall(r"\w+", s.lower())
    ta, tb = tok(a), tok(b); common = set(ta)&set(tb)
    if not ta and not tb: return 1.0
    if not common: return 0.0
    prec = sum(w in common for w in ta)/max(1,len(ta))
    rec  = sum(w in common for w in tb)/max(1,len(tb))
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--presets", type=str, default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--attack", type=str, default="homoglyph", choices=["homoglyph","synonym","trigger"])
    ap.add_argument("--random_precision", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token

    presets = json.loads(args.presets)
    model = SwitchableGPT2("gpt2", presets=presets)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    ds = load_dataset("rajpurkar/squad")["validation"].shuffle(seed=args.seed).select(range(args.n))
    rows = []; keys = list(presets.keys())
    for i, r in enumerate(ds):
        q = r["question"]
        if args.attack == "homoglyph": q2 = homoglyph_attack(q, ratio=0.2)
        elif args.attack == "synonym": q2 = simple_synonym_attack(q, ratio=0.3)
        else: q2 = q + TRIGGER_SUFFIX
        prompt = build_prompt(r["context"], q2) + " "
        truth = r["answers"]["text"][0] if r["answers"]["text"] else "unknown"
        if args.random_precision:
            model.random_preset(); used = model.active_preset
        else:
            used = keys[0]; model.set_preset(used)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=384-64)
        ids = enc["input_ids"].to(device); mask = enc["attention_mask"].to(device)
        out = model.inner.generate(input_ids=ids, attention_mask=mask,
                                   max_new_tokens=32, do_sample=False,
                                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
                                   no_repeat_ngram_size=3, repetition_penalty=1.1)
        gen = out[0][ids.shape[1]:]
        pred = tok.decode(gen, skip_special_tokens=True)
        rows.append({"i": i, "attack": args.attack, "preset": used, "pred": pred, "truth": truth,
                     "em": exact_match(pred, truth),
                     "f1": f1_score(pred, truth)})
    os.makedirs("results", exist_ok=True)
    outp = f"results/step6_{args.attack}_{'rand' if args.random_precision else 'fixed'}.json"
    with open(outp, "w") as f: json.dump(rows, f, indent=2)
    print(f"Wrote {outp} with {len(rows)} samples.")

if __name__ == "__main__":
    main()