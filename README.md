# GPT-2 fine-tuning and quantization on SQuAD

This project compares two LoRA training schedules on GPT-2 using the same three fake-quantization presets.

| Preset | Weight bits | Activation bits |
| --- | ---: | ---: |
| A | 8 | 8 |
| B | 4 | 4 |
| C | 3 | 4 |

Cascade distillation trains all presets on each batch and uses higher-precision logits to teach lower-precision presets. The distillation loss averages over next-token predictions for supervised answer tokens. Cyclic training selects one preset per batch on a cosine schedule. Each attention and MLP projection has one adapter per preset. The output head shares the original token embedding weights. The fake quantizer uses a scale per weight input feature and one scale for each activation tensor.

## Run the comparison

From the repository root, create a Python 3.12 environment. These CPU installation commands match the test workflow. For GPU training, install the matching PyTorch wheel for your CUDA runtime. The first experiment run downloads GPT-2 and SQuAD.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install transformers==4.57.6 datasets==4.0.0 pyyaml==6.0.3 pytest==9.2.0
python -m pytest -q
```

Both evaluators take the first 200 validation questions in dataset order and run every preset on each question. Use a separate output path to keep the archived experiment alongside a new run.

```bash
bash scripts/run_step3.sh --outdir runs/cascade
bash scripts/run_step4.sh --ckpt runs/cascade/switchable_gpt2.pt --out runs/cascade/validation.json
bash scripts/run_step5.sh --outdir runs/cyclic
bash scripts/run_step5_eval.sh --ckpt runs/cyclic/cyclic_gpt2.pt --out runs/cyclic/validation.json
```

Pass `--max_items` to change the sample size or `--preset B` to evaluate one preset. If training used a custom LoRA rank, pass the same `--rank` at evaluation. To run the optional robustness experiment on the cascade checkpoint, use `bash scripts/run_step6.sh --ckpt runs/cascade/switchable_gpt2.pt`.

## Reading the evaluation

The clean evaluator trims long contexts to preserve the question and answer cue. It scores the decoded continuation against every human reference and keeps the best exact match and token F1 for each question. Blank generations receive zero. Normalization follows the [SQuAD v1.1 scorer](https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py).

Each row records the SQuAD ID, preset, raw continuation, scored prediction, all references and scores. The sibling `*.summary.json` file contains the mean EM and F1 on a 0–1 scale, the count and the scoring method for each preset.

**Historical results.** The committed Step 4 and Step 5 outputs, related figures and report use an older evaluator. When its cleaned prediction was empty, it could select a context span using a reference answer. Those scores can overstate model performance. The archived rows omit the raw continuations, so corrected scores require rerunning generation. The training runs used 1,000 steps and generated answers can be incoherent.
