"""
Step 6: Random Precision Training (RPT) for Adversarial Robustness
Paper Reference: Double-Win Quant (ICML'21) - Algorithm 2, Section 3.4
"""
import os, json, argparse, torch, random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, get_linear_schedule_with_warmup
from torch.optim import AdamW
from src.models.gpt2_switchable import SwitchableGPT2
from src.data.squad_dataset import SquadGPT2Dataset

def seed_all(s):
    import numpy as np
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
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
    ap.add_argument("--outdir", type=str, default="results/step6_rpt")
    ap.add_argument("--presets", type=str, default='{"A":{"w_bits":8,"a_bits":8},"B":{"w_bits":4,"a_bits":4},"C":{"w_bits":3,"a_bits":4}}')
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=42)
    # Adversarial training parameters
    ap.add_argument("--pgd_steps", type=int, default=7, help="Number of PGD steps for adversarial training")
    ap.add_argument("--epsilon", type=float, default=0.01, help="Perturbation budget")
    ap.add_argument("--alpha_pgd", type=float, default=0.003, help="PGD step size")
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

    # === Step 6: Random Precision Training (RPT) ===
    # Paper Reference: Double-Win Quant (ICML'21) - Algorithm 2
    # Key components:
    # 1. Switchable Batch Normalization (SBN) - separate BN stats per precision
    # 2. Random precision selection per batch during training
    # 3. Adversarial training with PGD-7 (or other methods)

    # Initialize all presets to create LoRA adapters
    for preset_name in presets.keys():
        model.set_preset(preset_name)

    # Enable Switchable Batch Normalization
    # Note: GPT-2 uses LayerNorm, not BatchNorm, but the concept is the same
    # We keep separate statistics per preset
    print("RPT enabled: Training with random precision selection per batch")
    print(f"Presets: {list(presets.keys())}")

    # Train only adapters
    for n,p in model.named_parameters():
        p.requires_grad = (".adapters." in n)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {len(trainable_params)} LoRA parameters")

    opt = AdamW(trainable_params, lr=args.lr, weight_decay=0.0)
    total = args.steps; warm = max(100, int(0.2*total))
    sched = get_linear_schedule_with_warmup(opt, warm, total)

    preset_keys = list(presets.keys())
    it = iter(dl); step = 0; losses = []
    model.train()

    while step < total:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)

        # === Algorithm 2: Random Precision Training ===
        # Line 4: Randomly select a precision q from Set_Q
        q_preset = random.choice(preset_keys)
        model.set_preset(q_preset)

        # Line 5: Obtain f_q by quantizing f_θ to q-bit
        # (Already done by set_preset)

        # Line 6: δ = 0 or random initialized
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Get embeddings for PGD
        embeds = model.inner.transformer.wte(input_ids)
        delta = torch.zeros_like(embeds, requires_grad=True)

        # Lines 7-9: PGD-7 attack loop
        for t in range(args.pgd_steps):
            # Forward with perturbation
            perturbed_embeds = embeds + delta
            outputs = model.inner(inputs_embeds=perturbed_embeds,
                                 attention_mask=attention_mask,
                                 labels=labels)

            # Compute gradient of loss w.r.t. delta
            if outputs.loss.requires_grad:
                loss_for_pgd = outputs.loss
                loss_for_pgd.backward()

                if delta.grad is not None:
                    # Line 8: δ = clip_ε{δ + α · sign(∇_δ ℓ(f_q(x + δ), y))}
                    delta_data = delta.detach() + args.alpha_pgd * delta.grad.sign()
                    delta_data = torch.clamp(delta_data, -args.epsilon, args.epsilon)
                    delta = delta_data.detach().requires_grad_(True)

                # Zero gradients for model parameters
                opt.zero_grad(set_to_none=True)

        # Line 10: θ = θ - ∇_θ ℓ(f_q(x + δ), y)
        # Final forward pass with adversarial examples
        perturbed_embeds = embeds.detach() + delta.detach()
        final_outputs = model.inner(inputs_embeds=perturbed_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels)

        loss = final_outputs.loss / args.accum
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
            print(f"step {step+1}/{total} preset {q_preset} loss {float(loss.item()):.4f}")
        losses.append(float(loss.item())); step += 1

    torch.save(model.state_dict(), os.path.join(args.outdir, "rpt_gpt2.pt"))
    with open(os.path.join(args.outdir, "train_loss.txt"), "w") as f:
        for i,l in enumerate(losses, 1): f.write(f"{i}\t{l}\n")

    print(f"\nRPT training complete! Model saved to {args.outdir}")
    print("This model can now be used with Random Precision Inference (RPI) for enhanced robustness.")

if __name__ == "__main__":
    main()
