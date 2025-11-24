#!/usr/bin/env bash
set -e
python -m src.robustness.attacks --ckpt results/step3_switchable/switchable_gpt2.pt --attack homoglyph --random_precision --presets "$(cat configs/presets.yaml | python -c 'import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read())))')" "$@"
