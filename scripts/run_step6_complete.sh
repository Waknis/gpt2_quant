#!/usr/bin/env bash
# Step 6: Complete Adversarial Robustness Evaluation
# Tests Random Precision Inference (RPI) vs Fixed Precision
# Across three adversarial attacks: homoglyph, synonym, trigger

set -e

CKPT="results/step3_switchable/switchable_gpt2.pt"
PRESETS=$(cat configs/presets.yaml | python -c 'import sys, yaml, json; print(json.dumps(yaml.safe_load(sys.stdin.read())))')

echo "========================================="
echo "Step 6: Adversarial Robustness Evaluation"
echo "========================================="
echo ""
echo "Testing three attacks with two strategies:"
echo "  1. Fixed precision (baseline)"
echo "  2. Random precision (RPI - Random Precision Inference)"
echo ""

# Homoglyph Attack
echo "[1/6] Running homoglyph attack with FIXED precision..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack homoglyph \
  --presets "$PRESETS" \
  --n 150

echo "[2/6] Running homoglyph attack with RANDOM precision (RPI)..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack homoglyph \
  --random_precision \
  --presets "$PRESETS" \
  --n 150

# Synonym Attack
echo "[3/6] Running synonym attack with FIXED precision..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack synonym \
  --presets "$PRESETS" \
  --n 150

echo "[4/6] Running synonym attack with RANDOM precision (RPI)..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack synonym \
  --random_precision \
  --presets "$PRESETS" \
  --n 150

# Trigger Attack
echo "[5/6] Running trigger attack with FIXED precision..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack trigger \
  --presets "$PRESETS" \
  --n 150

echo "[6/6] Running trigger attack with RANDOM precision (RPI)..."
python -m src.robustness.attacks \
  --ckpt "$CKPT" \
  --attack trigger \
  --random_precision \
  --presets "$PRESETS" \
  --n 150

echo ""
echo "========================================="
echo "Step 6 Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved in results/ directory:"
echo "  - step6_homoglyph_fixed.json"
echo "  - step6_homoglyph_rand.json"
echo "  - step6_synonym_fixed.json"
echo "  - step6_synonym_rand.json"
echo "  - step6_trigger_fixed.json"
echo "  - step6_trigger_rand.json"
echo ""
echo "Next: Run analysis script to compare Fixed vs RPI performance"
