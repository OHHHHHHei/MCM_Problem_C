import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from data_processor import DataProcessor
from smc_inverse import SMCInverse, ModelParams

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="white", context="paper", font_scale=1.4)

def plot_forensic():
    # 1. Config & Initialize
    csv_path = '2026_MCM_Problem_C_Data.csv'
    target_season = 27
    target_contestant = "Bobby Bones"
    compare_contestant = "Milo Manheim" # The runner-up with high scores for contrast
    
    # Top-1 Best Parameters (from optimization results)
    best_params = ModelParams(
        n_particles=1000, # High precision for this specific plot
        rho=0.7,
        gamma=0.8,
        delta=1.8,
        alpha_discriminant=7.0,
        kappa=0.2,
        beta_judge=2.0
    )
    
    print(f"Running model for Season {target_season} with params: {best_params}...")
    
    dp = DataProcessor(csv_path)
    model = SMCInverse(dp, best_params)
    
    # 2. Run Single Season
    result = model.run_season(target_season, verbose=False)
    
    # 3. Extract Time Series Data
    weeks = []
    bobby_judge_norm = [] # Normalized judge score
    bobby_share = []
    
    milo_judge_norm = []
    milo_share = []
    
    estimates = result['weekly_estimates']
    # Get all active weeks for this season
    all_weeks = sorted(estimates.keys())
    
    for w in all_weeks:
        # Get active contestants to normalize judge scores properly
        active = dp.get_active_contestants(target_season, w)
        active_names = [c.name for c in active]
        
        if target_contestant not in active_names:
            continue
            
        weeks.append(w)
        
        # Judge Scores
        judge_scores = {c.name: dp.get_weekly_total_score(c, w) for c in active}
        total_j = sum(judge_scores.values())
        
        bobby_j = judge_scores.get(target_contestant, 0) / total_j if total_j else 0
        milo_j = judge_scores.get(compare_contestant, 0) / total_j if total_j else 0
        
        bobby_judge_norm.append(bobby_j)
        milo_judge_norm.append(milo_j)
        
        # Model Shares
        week_est = estimates[w]
        bobby_s = next((e['mean'] for n, e in week_est.items() if n == target_contestant), 0)
        milo_s = next((e['mean'] for n, e in week_est.items() if n == compare_contestant), 0)
        
        bobby_share.append(bobby_s)
        milo_share.append(milo_s)

    # 4. Plotting
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)
    
    x = np.array(weeks)
    width = 0.35
    
    # Left Axis: Judge Scores (Bar)
    # Showing Bobby vs Milo contrast in bars is crowded, let's just show Bobby's low bars vs Global Avg? 
    # Or just Bobby's Bars. The user asked for "Bobby's bars are short".
    
    ax1.set_xlabel('Competition Week', fontsize=14, labelpad=10)
    ax1.set_ylabel('Normalized Judge Score (Official)', color='gray', fontsize=14)
    
    bars = ax1.bar(x, bobby_judge_norm, width, color='gray', alpha=0.4, label=f'{target_contestant} (Judge Score)')
    ax1.tick_params(axis='y', labelcolor='gray')
    ax1.set_ylim(0, 0.25) # Judge scores usually around 1/N, max 25% is safe for normalized
    
    # Add value labels on bars
    # for bar in bars:
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., height,
    #             f'{height:.1%}', ha='center', va='bottom', fontsize=9, color='gray')

    # Right Axis: Vote Shares (Line)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Estimated Fan Vote Share (Hidden)', color='#d62728', fontsize=14)
    
    # Plot Bobby (The Populist)
    line1 = ax2.plot(x, bobby_share, marker='D', color='#d62728', linewidth=3, markersize=8, label=f'{target_contestant} (Fan Share)')
    
    # Plot Milo (The Technician) for contrast
    line2 = ax2.plot(x, milo_share, marker='o', color='#1f77b4', linewidth=2, linestyle='--', alpha=0.7, label=f'{compare_contestant} (Fan Share)')

    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(0, 0.6) # Shares can go high
    
    # Legend
    # Combine bars and lines
    lines = [bars] + line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=12, frameon=True)
    
    # Title & Story
    plt.title(f'Forensic Analysis of Season {target_season}: The "Populist Uprising"\nModel captures divergence between low judge scores and skyrocketing popularity', 
              fontsize=16, weight='bold', pad=20)

    # Annotate the gap
    # Find the week with biggest gap
    gap_idx = np.argmax(np.array(bobby_share) - np.array(bobby_judge_norm))
    gap_week = weeks[gap_idx]
    gap_val = bobby_share[gap_idx]
    
    ax2.annotate(f'Massive Popularity Gap\n(Share={gap_val:.1%} vs Judge={bobby_judge_norm[gap_idx]:.1%})', 
                 xy=(gap_week, gap_val), 
                 xytext=(gap_week, gap_val + 0.1),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, weight='bold', ha='center')

    ax1.grid(False)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    output_path = 'output/forensic_analysis_bobby.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_forensic()
