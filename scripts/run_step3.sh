#!/usr/bin/env bash
set -e
python -m src.training.train_switchable --outdir results/step3_switchable --presets "$(cat configs/presets.yaml | python -c 'import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read())))')" "$@"
