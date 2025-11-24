#!/usr/bin/env bash
set -e
python -m src.training.eval_squad --ckpt results/step3_switchable/switchable_gpt2.pt --presets "$(cat configs/presets.yaml | python -c 'import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read())))')" "$@"
