import os, json, argparse, torch, math
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.models.gpt2_switchable import SwitchableGPT2
from src.data.squad_dataset import SquadGPT2Dataset

def seed_all(s):
    import numpy as np, random, torch
    import random as pyrand
    pyrand.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def make_collate(tokenizer):
    pad_id = tokenizer.eos_token_id
    def collate(samples):
        B = len(samples)
        ids_list, labs_list = [], []
        for s in samples:
            ids = s["prompt_ids"] + s["answer_ids"]
            labs = [-100]*len(s["prompt_ids"]) + s["answer_ids"]
            ids_list.append(ids); labs_list.append(labs)
        maxlen = max(len(x) for x in ids_list)
        input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, maxlen), dtype=torch.long)
        labels = torch.full((B, maxlen), -100, dtype=torch.long)
        for i,(ids,labs) in enumerate(zip(ids_list, labs_list)):
            L = len(ids)
            input_ids[i,:L] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i,:L] = 1
            labels[i,:L] = torch.tensor(labs, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results/step5_cyclic")
    # === Step 5: Cyclic Precision Training (CPT) ===
    # Paper Reference: CPT (ICLR'21) - Section 3.2, Eq. (1)
    # Cosine schedule: B_t^n = ⌈B_min^n + 0.5(B_max^n - B_min^n)(1 - cos(πt % T_n / T_n))⌉
    ap.add_argument("--num_cycles", type=int, default=32, help="Number of cyclic precision cycles (N in paper, default=32)")
    ap.add_argument("--total_steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--presets", type=str, default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "hparams.json"), "w") as f: json.dump(vars(args), f, indent=2)

    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    ds = SquadGPT2Dataset(split="train", max_samples=None, max_len=args.max_len, tokenizer=tok, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=make_collate(tok))

    presets = json.loads(args.presets)
    model = SwitchableGPT2("gpt2", presets=presets, lora_rank=args.rank, lora_alpha=args.alpha).to(device)
    model.inner.resize_token_embeddings(len(tok))

    # Initialize all presets to create LoRA adapters
    for preset_name in presets.keys():
        model.set_preset(preset_name)

    for n,p in model.named_parameters():
        p.requires_grad = (".adapters." in n)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(trainable_params)} LoRA parameters")

    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1*args.total_steps), args.total_steps)

    # === CPT Implementation: Cosine Schedule ===
    # Sort presets by bit-width to get B_min and B_max
    def preset_score(k):
        cfg = presets[k]
        return int(cfg.get("w_bits", 8)) + int(cfg.get("a_bits", 8))

    preset_keys = sorted(presets.keys(), key=preset_score)
    B_min_key = preset_keys[0]   # Lowest bit-width preset
    B_max_key = preset_keys[-1]  # Highest bit-width preset

    # Cycle length T_n for each of the N cycles
    T_n = args.total_steps / args.num_cycles

    print(f"CPT Config: {args.num_cycles} cycles, T_n={T_n:.1f} steps/cycle")
    print(f"  B_min={B_min_key} ({preset_score(B_min_key)}), B_max={B_max_key} ({preset_score(B_max_key)})")

    it = iter(dl); step = 0
    while step < args.total_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        # === CPT Cosine Schedule (Eq. 1 from paper) ===
        # B_t^n = ⌈B_min^n + 0.5(B_max^n - B_min^n)(1 - cos(πt % T_n / T_n))⌉
        # Map cosine value [0, 1] to preset indices
        t_mod = step % T_n  # Position within current cycle
        cos_val = (1 - math.cos(math.pi * t_mod / T_n)) / 2  # Ranges from 0 to 1

        # Interpolate between min and max preset indices
        preset_idx = round(cos_val * (len(preset_keys) - 1))
        preset = preset_keys[preset_idx]

        model.set_preset(preset)
        out = model.inner(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device),
                          labels=batch["labels"].to(device))
        loss = out.loss / args.accum
        if not torch.isfinite(loss):
            print("[skip] non-finite loss"); step += 1; continue
        loss.backward()

        if (step+1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = None
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if (step+1) % 20 == 0:
            cycle_num = int(step / T_n) + 1
            print(f"step {step+1}/{args.total_steps} cycle {cycle_num}/{args.num_cycles} preset {preset} loss {float(loss.item()):.4f}")
        step += 1

    torch.save(model.state_dict(), os.path.join(args.outdir, "cyclic_gpt2.pt"))

if __name__ == "__main__":
    main()