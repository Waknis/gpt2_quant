#!/usr/bin/env python3
"""
Generate comprehensive analysis and visualizations for academic report
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for academic figures
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

def load_json(path):
    """Load JSON file"""
    with open(path) as f:
        return json.load(f)

def create_step4_comparison():
    """Step 4: Quantization bit-width comparison"""
    data = load_json('results/step4_eval.summary.json')

    presets = list(data.keys())
    preset_labels = {
        'A': 'W8A8',
        'B': 'W4A4',
        'C': 'W3A4'
    }

    em_scores = [data[p]['em'] * 100 for p in presets]
    f1_scores = [data[p]['f1'] * 100 for p in presets]

    # Calculate theoretical BitOPs (relative to 8-bit baseline)
    bitops = {
        'A': 1.0,      # 8x8 = 64 bit-ops
        'B': 0.25,     # 4x4 = 16 bit-ops (4x reduction)
        'C': 0.1875    # 3x4 = 12 bit-ops (5.3x reduction)
    }
    bitops_values = [bitops[p] for p in presets]

    bitops_values = [bitops[p] for p in presets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # Plot 1: Exact Match
    ax = axes[0]
    bars = ax.bar(range(len(presets)), em_scores, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quantization Configuration', fontsize=11, fontweight='bold')
    ax.set_title('(a) Exact Match Accuracy', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels([preset_labels[p] for p in presets], fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(em_scores) * 1.4)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, em_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 2: F1 Score
    ax = axes[1]
    bars = ax.bar(range(len(presets)), f1_scores, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
    ax.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quantization Configuration', fontsize=11, fontweight='bold')
    ax.set_title('(b) F1 Score', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(presets)))
    ax.set_xticklabels([preset_labels[p] for p in presets], fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(f1_scores) * 1.4)

    for i, (bar, val) in enumerate(zip(bars, f1_scores)):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, f'{val:.1f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Accuracy-Efficiency Trade-off
    ax = axes[2]
    ax.scatter(bitops_values, f1_scores, s=300, c=['#2ecc71', '#3498db', '#e74c3c'],
               alpha=0.7, edgecolors='black', linewidths=2)

    for i, preset in enumerate(presets):
        ax.annotate(preset_labels[preset].split('\n')[0],
                   (bitops_values[i], f1_scores[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    ax.set_xlabel('Relative BitOPs (normalized to W8A8)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax.set_title('(c) Accuracy-Efficiency Trade-off', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1.1)

    # plt.tight_layout() # Using constrained_layout instead
    plt.savefig('results/figures/step4_quantization_comparison.png', dpi=300, bbox_inches='tight')
    print("Created: results/figures/step4_quantization_comparison.png")

    return data

def create_step5_comparison():
    """Step 5: Cascade vs Cyclic training comparison"""
    step3_data = load_json('results/step4_eval.summary.json')  # Cascade training
    step5_data = load_json('results/step5_eval.summary.json')  # Cyclic training

    presets = list(step3_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Prepare data
    cascade_em = [step3_data[p]['em'] * 100 for p in presets]
    cascade_f1 = [step3_data[p]['f1'] * 100 for p in presets]
    cyclic_em = [step5_data[p]['em'] * 100 for p in presets]
    cyclic_f1 = [step5_data[p]['f1'] * 100 for p in presets]

    x = np.arange(len(presets))
    width = 0.35

    # Plot 1: Exact Match Comparison
    ax = axes[0]
    bars1 = ax.bar(x - width/2, cascade_em, width, label='Cascade Distillation (Step 3)',
                   color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, cyclic_em, width, label='Cyclic Precision (Step 5)',
                   color='#e74c3c', alpha=0.7)

    ax.set_ylabel('Exact Match (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quantization Configuration', fontsize=11, fontweight='bold')
    ax.set_title('(a) EM: Cascade vs Cyclic Training', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['W8A8\n(A)', 'W4A4\n(B)', 'W3A4\n(C)'], fontsize=10)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # Plot 2: F1 Score Comparison
    ax = axes[1]
    bars1 = ax.bar(x - width/2, cascade_f1, width, label='Cascade Distillation (Step 3)',
                   color='#3498db', alpha=0.7)
    bars2 = ax.bar(x + width/2, cyclic_f1, width, label='Cyclic Precision (Step 5)',
                   color='#e74c3c', alpha=0.7)

    ax.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quantization Configuration', fontsize=11, fontweight='bold')
    ax.set_title('(b) F1: Cascade vs Cyclic Training', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['W8A8\n(A)', 'W4A4\n(B)', 'W3A4\n(C)'], fontsize=10)
    ax.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=8)

    # plt.tight_layout() # Using constrained_layout instead
    plt.savefig('results/figures/step5_cascade_vs_cyclic.png', dpi=300, bbox_inches='tight')
    print(" Created: results/figures/step5_cascade_vs_cyclic.png")

    # Compute improvements
    improvements = {}
    for p in presets:
        improvements[p] = {
            'em_diff': (step5_data[p]['em'] - step3_data[p]['em']) * 100,
            'f1_diff': (step5_data[p]['f1'] - step3_data[p]['f1']) * 100
        }

    return step3_data, step5_data, improvements

def create_step6_comparison():
    """Step 6: Adversarial robustness with RPI"""
    data = load_json('results/step6_summary.json')

    attacks = ['homoglyph', 'synonym', 'trigger']
    attack_labels = {
        'homoglyph': 'Homoglyph\n(Visual similarity)',
        'synonym': 'Synonym\n(Semantic paraphrase)',
        'trigger': 'Trigger Suffix\n(Prompt injection)'
    }

    fixed_f1 = [data[a]['fixed']['f1'] for a in attacks]
    rand_f1 = [data[a]['rand']['f1'] for a in attacks]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # Plot 1: Fixed vs Random Precision
    ax = axes[0]
    x = np.arange(len(attacks))
    width = 0.35

    bars1 = ax.bar(x - width/2, fixed_f1, width, label='Fixed Precision',
                   color='#95a5a6', alpha=0.7)
    bars2 = ax.bar(x + width/2, rand_f1, width, label='Random Precision (RPI)',
                   color='#2ecc71', alpha=0.7)

    ax.set_ylabel('F1 Score (%)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Adversarial Attack Type', fontsize=11, fontweight='bold')
    ax.set_title('(a) Fixed vs Random Precision Inference', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([attack_labels[a] for a in attacks], fontsize=10)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    
    # Increase ylim to prevent overlap with title
    ax.set_ylim(0, max(max(fixed_f1), max(rand_f1)) * 1.4)

    # Plot 2: Improvement from RPI
    ax = axes[1]
    improvements = [rand_f1[i] - fixed_f1[i] for i in range(len(attacks))]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    bars = ax.bar(range(len(attacks)), improvements, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_ylabel('F1 Score Improvement (pp)', fontsize=11, fontweight='bold')
    ax.set_xlabel('Adversarial Attack Type', fontsize=11, fontweight='bold')
    ax.set_title('(b) RPI Robustness Improvement', fontsize=12, fontweight='bold')
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels([attack_labels[a] for a in attacks], fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height > 0 else -0.1),
               f'{imp:+.2f}pp', ha='center', va='bottom' if height > 0 else 'top',
               fontsize=9, fontweight='bold')

    # Increase ylim to prevent overlap with title
    max_imp = max(abs(min(improvements)), abs(max(improvements)))
    ax.set_ylim(min(improvements) * 1.5 if min(improvements) < 0 else 0, max(improvements) * 1.6)

    # plt.tight_layout() # Using constrained_layout instead
    plt.savefig('results/figures/step6_adversarial_robustness.png', dpi=300, bbox_inches='tight')
    print("Created: results/figures/step6_adversarial_robustness.png")

    return data, improvements

def create_summary_table():
    """Create comprehensive summary table"""
    step3 = load_json('results/step4_eval.summary.json')
    step5 = load_json('results/step5_eval.summary.json')
    step6 = load_json('results/step6_summary.json')

    print("\n" + "="*100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*100)

    print("\n[STEP 4] Switchable Quantization (Cascade Distillation Training)")
    print("-"*100)
    print(f"{'Preset':<10} {'Config':<20} {'EM (%)':<12} {'F1 (%)':<12} {'BitOPs':<15}")
    print("-"*100)
    print(f"{'A':<10} {'W8A8':<20} {step3['A']['em']*100:<12.2f} {step3['A']['f1']*100:<12.2f} {'1.0x (baseline)':<15}")
    print(f"{'B':<10} {'W4A4':<20} {step3['B']['em']*100:<12.2f} {step3['B']['f1']*100:<12.2f} {'0.25x (4x faster)':<15}")
    print(f"{'C':<10} {'W3A4':<20} {step3['C']['em']*100:<12.2f} {step3['C']['f1']*100:<12.2f} {'0.19x (5.3x faster)':<15}")

    print("\n[STEP 5] Cyclic Precision Training")
    print("-"*100)
    print(f"{'Preset':<10} {'Config':<20} {'EM (%)':<12} {'F1 (%)':<12} {'vs Cascade':<15}")
    print("-"*100)
    for p in ['A', 'B', 'C']:
        em_diff = (step5[p]['em'] - step3[p]['em']) * 100
        f1_diff = (step5[p]['f1'] - step3[p]['f1']) * 100
        print(f"{p:<10} {'W8A8' if p=='A' else 'W4A4' if p=='B' else 'W3A4':<20} "
              f"{step5[p]['em']*100:<12.2f} {step5[p]['f1']*100:<12.2f} "
              f"EM:{em_diff:+.1f} F1:{f1_diff:+.1f}")

    print("\n[STEP 6] Adversarial Robustness (Random Precision Inference)")
    print("-"*100)
    print(f"{'Attack':<15} {'Fixed F1 (%)':<15} {'RPI F1 (%)':<15} {'Improvement':<15}")
    print("-"*100)
    for attack in ['homoglyph', 'synonym', 'trigger']:
        fixed = step6[attack]['fixed']['f1']
        rand = step6[attack]['rand']['f1']
        imp = rand - fixed
        print(f"{attack.capitalize():<15} {fixed:<15.2f} {rand:<15.2f} {imp:+.2f}pp")

    avg_fixed = np.mean([step6[a]['fixed']['f1'] for a in ['homoglyph', 'synonym', 'trigger']])
    avg_rand = np.mean([step6[a]['rand']['f1'] for a in ['homoglyph', 'synonym', 'trigger']])
    print("-"*100)
    print(f"{'Average':<15} {avg_fixed:<15.2f} {avg_rand:<15.2f} {avg_rand-avg_fixed:+.2f}pp")

    print("\n" + "="*100)

def main():
    """Generate all visualizations and analysis"""
    # Create output directory
    Path('results/figures').mkdir(parents=True, exist_ok=True)

    print("\n" + "="*100)
    print("GENERATING COMPREHENSIVE REPORT DATA AND VISUALIZATIONS")
    print("="*100 + "\n")

    print("[1/4] Analyzing Step 4: Quantization bit-width configurations...")
    step4_data = create_step4_comparison()

    print("[2/4] Analyzing Step 5: Cascade vs Cyclic training...")
    step3_data, step5_data, improvements = create_step5_comparison()

    print("[3/4] Analyzing Step 6: Adversarial robustness...")
    step6_data, step6_improvements = create_step6_comparison()

    print("[4/4] Creating summary tables...")
    create_summary_table()

    print("\n" + "="*100)
    print("ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("="*100)
    print("\nGenerated files:")
    print("  results/figures/step4_quantization_comparison.png")
    print("  results/figures/step5_cascade_vs_cyclic.png")
    print("  results/figures/step6_adversarial_robustness.png")
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
