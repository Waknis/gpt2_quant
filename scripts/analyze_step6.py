#!/usr/bin/env python3
"""
Step 6: Analyze Adversarial Robustness Results
Compares Fixed Precision vs Random Precision Inference (RPI)
"""
import json
import os
from collections import defaultdict

def load_results(attack, strategy):
    """Load results file for given attack and strategy"""
    filename = f"results/step6_{attack}_{strategy}.json"
    if not os.path.exists(filename):
        return None
    with open(filename) as f:
        return json.load(f)

def compute_metrics(data):
    """Compute average EM and F1 scores"""
    if not data:
        return {"em": 0.0, "f1": 0.0, "count": 0}

    em_scores = [item["em"] for item in data]
    f1_scores = [item["f1"] for item in data]

    return {
        "em": sum(em_scores) / len(em_scores) * 100,  # Convert to percentage
        "f1": sum(f1_scores) / len(f1_scores) * 100,
        "count": len(data)
    }

def analyze_per_preset(data):
    """Analyze results per preset (only for RPI)"""
    if not data:
        return {}

    by_preset = defaultdict(list)
    for item in data:
        by_preset[item["preset"]].append(item)

    results = {}
    for preset, items in by_preset.items():
        results[preset] = compute_metrics(items)

    return results

def print_comparison_table():
    """Print comparison table for all attacks and strategies"""

    attacks = ["homoglyph", "synonym", "trigger"]
    strategies = ["fixed", "rand"]

    print("=" * 100)
    print("STEP 6: ADVERSARIAL ROBUSTNESS EVALUATION - Random Precision Inference (RPI)")
    print("=" * 100)
    print("\nPaper Reference: Double-Win Quant (ICML'21) - Algorithm 1 (RPI)")
    print("\nQuestion: Does random precision switching at inference time improve adversarial robustness?")
    print("\n" + "=" * 100)

    for attack in attacks:
        print(f"\n{'=' * 100}")
        print(f"ATTACK: {attack.upper()}")
        print(f"{'=' * 100}")

        results = {}
        for strategy in strategies:
            data = load_results(attack, strategy)
            if data:
                results[strategy] = compute_metrics(data)

        if not results:
            print(f"  No results found for {attack} attack")
            continue

        # Print comparison
        print(f"\n{'Strategy':<25} {'Exact Match (EM)':<20} {'F1 Score':<20} {'Samples':<10}")
        print("-" * 100)

        for strategy in strategies:
            if strategy in results:
                metrics = results[strategy]
                label = "Fixed Precision" if strategy == "fixed" else "Random Precision (RPI)"
                print(f"{label:<25} {metrics['em']:>6.2f}%{' ':>13} {metrics['f1']:>6.2f}%{' ':>13} {metrics['count']:<10}")

        # Calculate improvement
        if "fixed" in results and "rand" in results:
            em_improvement = results["rand"]["em"] - results["fixed"]["em"]
            f1_improvement = results["rand"]["f1"] - results["fixed"]["f1"]

            print("-" * 100)
            print(f"{'RPI Improvement':<25} {em_improvement:>+6.2f}pp{' ':>12} {f1_improvement:>+6.2f}pp")

            if em_improvement > 0 or f1_improvement > 0:
                print(f"\n RPI IMPROVES robustness against {attack} attack!")
            elif em_improvement < 0 or f1_improvement < 0:
                print(f"\n RPI DECREASES robustness against {attack} attack")
            else:
                print(f"\n RPI has NO EFFECT on robustness against {attack} attack")

        # Show per-preset breakdown for RPI
        rand_data = load_results(attack, "rand")
        if rand_data:
            preset_metrics = analyze_per_preset(rand_data)
            if len(preset_metrics) > 1:  # Only show if multiple presets
                print(f"\nRPI Per-Preset Breakdown:")
                print(f"  {'Preset':<10} {'EM':<10} {'F1':<10} {'Count':<10}")
                print("  " + "-" * 40)
                for preset, metrics in sorted(preset_metrics.items()):
                    print(f"  {preset:<10} {metrics['em']:>6.2f}%   {metrics['f1']:>6.2f}%   {metrics['count']:<10}")

    # Overall summary
    print(f"\n{'=' * 100}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 100}\n")

    overall_fixed_em = []
    overall_fixed_f1 = []
    overall_rand_em = []
    overall_rand_f1 = []

    for attack in attacks:
        fixed_data = load_results(attack, "fixed")
        rand_data = load_results(attack, "rand")

        if fixed_data:
            metrics = compute_metrics(fixed_data)
            overall_fixed_em.append(metrics["em"])
            overall_fixed_f1.append(metrics["f1"])

        if rand_data:
            metrics = compute_metrics(rand_data)
            overall_rand_em.append(metrics["em"])
            overall_rand_f1.append(metrics["f1"])

    if overall_fixed_em and overall_rand_em:
        avg_fixed_em = sum(overall_fixed_em) / len(overall_fixed_em)
        avg_fixed_f1 = sum(overall_fixed_f1) / len(overall_fixed_f1)
        avg_rand_em = sum(overall_rand_em) / len(overall_rand_em)
        avg_rand_f1 = sum(overall_rand_f1) / len(overall_rand_f1)

        print(f"Average across all attacks:")
        print(f"  Fixed Precision:        EM={avg_fixed_em:.2f}%  F1={avg_fixed_f1:.2f}%")
        print(f"  Random Precision (RPI): EM={avg_rand_em:.2f}%  F1={avg_rand_f1:.2f}%")
        print(f"  Improvement:            EM={avg_rand_em - avg_fixed_em:+.2f}pp F1={avg_rand_f1 - avg_fixed_f1:+.2f}pp")

        print(f"\n{'=' * 100}")
        print("CONCLUSION")
        print(f"{'=' * 100}\n")

        if avg_rand_em > avg_fixed_em or avg_rand_f1 > avg_fixed_f1:
            print(" Random Precision Inference (RPI) IMPROVES adversarial robustness on average!")
            print("\nThis aligns with the Double-Win Quant (ICML'21) paper findings:")
            print("  - Adversarial examples have poor transferability across different precisions")
            print("  - Random precision switching at inference acts as a defense mechanism")
        elif avg_rand_em < avg_fixed_em or avg_rand_f1 < avg_fixed_f1:
            print(" Random Precision Inference (RPI) DECREASES adversarial robustness on average")
            print("\nPotential reasons for divergence from paper:")
            print("  - LLMs may behave differently than CNNs (paper focused on vision tasks)")
            print("  - Different attack types (text vs image perturbations)")
            print("  - Model architecture differences (Transformer vs CNN)")
        else:
            print(" Random Precision Inference (RPI) has NO SIGNIFICANT EFFECT")

    print(f"\n{'=' * 100}\n")

def save_summary():
    """Save summary statistics to JSON"""
    attacks = ["homoglyph", "synonym", "trigger"]
    summary = {}

    for attack in attacks:
        summary[attack] = {}
        for strategy in ["fixed", "rand"]:
            data = load_results(attack, strategy)
            if data:
                summary[attack][strategy] = compute_metrics(data)

    with open("results/step6_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Summary saved to: results/step6_summary.json")

if __name__ == "__main__":
    print_comparison_table()
    save_summary()
