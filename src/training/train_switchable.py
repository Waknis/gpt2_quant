import os, json, argparse, torch, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.models.gpt2_switchable import SwitchableGPT2
from src.data.squad_dataset import SquadGPT2Dataset

def seed_all(s):
    import numpy as np, random, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def make_collate(tokenizer):
    pad_id = tokenizer.eos_token_id  # eos as pad; we will always pass attention_mask
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
    ap.add_argument("--outdir", type=str, default="results/step3_switchable")
    ap.add_argument("--presets", type=str, default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--beta_cdt", type=float, default=0.02)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.outdir, exist_ok=True)
    with open(os.path.join(args.outdir, "hparams.json"), "w") as f: json.dump(vars(args), f, indent=2)

    tok = GPT2TokenizerFast.from_pretrained("gpt2"); tok.pad_token = tok.eos_token
    max_samples = None if not args.fast else 400
    ds = SquadGPT2Dataset(split="train", max_samples=max_samples, max_len=args.max_len, tokenizer=tok, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=make_collate(tok))

    presets = json.loads(args.presets)
    model = SwitchableGPT2("gpt2", presets=presets, lora_rank=args.rank, lora_alpha=args.alpha).to(device)
    model.inner.resize_token_embeddings(len(tok))

    # Initialize all presets to create LoRA adapters
    for preset_name in presets.keys():
        model.set_preset(preset_name)

    # Train only adapters
    for n,p in model.named_parameters():
        p.requires_grad = (".adapters." in n)

    # Verify we have trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found! Check LoRA adapter initialization.")
    print(f"Training {len(trainable_params)} LoRA parameters")

    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    total = args.steps; warm = max(100, int(0.2*total))
    sched = get_linear_schedule_with_warmup(opt, warm, total)

    keys = list(presets.keys()); it = iter(dl); step = 0; losses = []

    # === Step 3: Switchable Precision Training ===
    # Paper Reference: InstantNet (ICLR'21) - Section III.B, Eq. (1)
    # "Cascade Distillation Training (CDT)"
    # L_total = (1/N) Σ L_cas_train(Q_i(ω))
    # where L_cas_train(Q_i(ω)) = L_ce(Q_i(ω), label) + β Σ_{j=i+1}^{N-1} L_mse(Q_i(ω), SG(Q_j(ω)))
    # Key insight: Each preset distills from ALL higher bit-widths, not just the highest

    # Sort presets by bit-width (lowest to highest)
    def preset_score(k):
        cfg = presets[k]
        return int(cfg.get("w_bits", 8)) + int(cfg.get("a_bits", 8))
    keys_sorted = sorted(keys, key=preset_score)  # A, B, C where A < B < C
    preset_bits = {k: preset_score(k) for k in keys}

    model.train()
    while step < total:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        loss_sum = 0.0
        T = args.temperature  # Temperature for knowledge distillation

        # === Cascade Distillation: Forward all presets and store outputs ===
        preset_outputs = {}
        for preset in keys_sorted:
            model.set_preset(preset)
            output = model.inner(input_ids=batch["input_ids"].to(device),
                                attention_mask=batch["attention_mask"].to(device),
                                labels=batch["labels"].to(device))
            preset_outputs[preset] = output

        # === Compute loss for each preset with cascade distillation ===
        for preset in keys_sorted:
            output = preset_outputs[preset]
            current_bits = preset_bits[preset]

            # Cross-entropy loss (always present)
            loss = output.loss

            # Distill from ALL higher bit-widths (cascade)
            kd_loss = 0.0
            num_higher = 0
            for other_preset in keys_sorted:
                other_bits = preset_bits[other_preset]
                if other_bits > current_bits:
                    # Temperature-scaled KL divergence from higher bit-width
                    # SG(Q_j(ω)) = stop gradient on teacher logits
                    other_output = preset_outputs[other_preset]
                    t_p = F.softmax(other_output.logits.detach() / T, dim=-1)
                    s_logp = F.log_softmax(output.logits / T, dim=-1)
                    kd_loss += F.kl_div(s_logp, t_p, reduction="batchmean") * (T * T)
                    num_higher += 1

            # Average distillation loss from all higher bit-widths
            if num_higher > 0:
                kd_loss = kd_loss / num_higher
                loss = loss + args.beta_cdt * kd_loss

            # Backpropagate
            loss = (loss / len(keys)) / args.accum
            if not torch.isfinite(loss):
                print(f"[skip] non-finite loss for preset {preset}"); continue
            loss.backward(); loss_sum += float(loss.item())

        if (step+1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    p.grad = None
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)

        if (step+1) % 20 == 0:
            denom = max(1, len(keys))
            print(f"step {step+1} loss {(loss_sum/denom):.4f}")
        losses.append(loss_sum); step += 1

    torch.save(model.state_dict(), os.path.join(args.outdir, "switchable_gpt2.pt"))
    with open(os.path.join(args.outdir, "train_loss.txt"), "w") as f:
        for i,l in enumerate(losses, 1): f.write(f"{i}\t{l}\n")

if __name__ == "__main__":
    main()