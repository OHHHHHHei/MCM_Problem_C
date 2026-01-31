import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def plot_fan_independence():
    input_file = 'output/optimization_overnight_results.csv'
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # Recalculate Weighted Score with user's preferred weights if needed, 
    # but here we just need to select the "Best Representative" for each Gamma.
    # We select the model that performs BEST overall for each Gamma level 
    # to show the "theoretical limit" of that hypothesis.
    
    if 'weighted_score' not in df.columns:
         df['weighted_score'] = df['top3_hit'] * 0.6 + df['map_match_rate'] * 0.2 + df['posterior_consistency'] * 0.2

    # Group by Gamma and take the max of metrics directly? 
    # Better: Pick the best row (by weighted score) for each gamma, 
    # then report that row's Top-3 Hit and Consistency.
    # This prevents mixing metrics from different parameter sets.
    
    idx = df.groupby('gamma')['weighted_score'].idxmax()
    best_per_gamma = df.loc[idx].sort_values(by='gamma')
    
    print("\nBest configurations per Gamma:")
    print(best_per_gamma[['gamma', 'top3_hit', 'posterior_consistency', 'weighted_score']].to_string(index=False))

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # X-axis
    gammas = best_per_gamma['gamma']
    
    # Y-axis 1: Top-3 Hit Rate
    color1 = '#1f77b4' # Muted Blue
    ax1.set_xlabel('Judge Influence Factor ($\gamma$)', fontsize=14, labelpad=10)
    ax1.set_ylabel('Top-3 Hit Rate (Accuracy)', color=color1, fontsize=14)
    line1 = ax1.plot(gammas, best_per_gamma['top3_hit'], marker='o', color=color1, linewidth=2.5, markersize=8, label='Top-3 Hit Rate')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Shade the area under the curve slightly
    ax1.fill_between(gammas, best_per_gamma['top3_hit'], alpha=0.1, color=color1)

    # Y-axis 2: Posterior Consistency
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color2 = '#d62728' # Muted Red
    ax2.set_ylabel('Posterior Consistency (Robustness)', color=color2, fontsize=14)  
    line2 = ax2.plot(gammas, best_per_gamma['posterior_consistency'], marker='s', color=color2, linewidth=2.5, markersize=8, linestyle='--', label='Posterior Consistency')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower left', fontsize=12, frameon=True, framealpha=0.9)

    # Title and annotations
    plt.title('Evidence of Fan Independence:\nModel Performance Declines as Judge Influence Increases', fontsize=16, weight='bold', pad=20)
    
    # Annotate the peak
    peak_gamma = best_per_gamma.iloc[0]['gamma']
    peak_acc = best_per_gamma.iloc[0]['top3_hit']
    if peak_gamma == 0:
        ax1.annotate(f'Peak Accuracy: {peak_acc:.1%}\n(Independent Fans)', 
                     xy=(peak_gamma, peak_acc), 
                     xytext=(peak_gamma + 0.15, peak_acc - 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                     fontsize=11, weight='bold')

    # Grid
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax2.grid(False) # Turn off second grid to avoid clutter

    # Save
    output_path = 'output/fan_independence_analysis.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_fan_independence()
