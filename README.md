# GPT2 Switchable & Dynamic Quantization (M2)

This model was trained for only 1,000 steps. It is a proof-of-concept and has not converged, so generated outputs may be low quality or incoherent.

- QAT-LLM style symmetric MinMax quant (weights per-channel, activations per-token).
- LoRA on all linear layers, one adapter per precision preset.
- Joint multi-precision training (Step 3), cyclic precision training (Step 5), random-precision inference + robustness (Step 6).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio transformers datasets textattack pyyaml
```

## Run
```bash
bash scripts/run_step3.sh
bash scripts/run_step4.sh #runs step 3 eval
bash scripts/run_step5.sh
bash scripts/run_step5_eval.sh #runs step 5 Squad eval
bash scripts/run_step6.sh
```
