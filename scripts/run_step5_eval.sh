#!/usr/bin/env bash
set -e
python -m src.training.eval_squad --ckpt results/step5_cyclic/cyclic_gpt2.pt --out results/step5_eval.json --presets "$(cat configs/presets.yaml | python -c 'import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read())))')" "$@"
