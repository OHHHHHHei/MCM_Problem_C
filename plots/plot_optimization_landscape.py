import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_landscape():
    # Adjust path to output file
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_overnight_results.csv'))
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print("Loading data...")
    df = pd.read_csv(input_file)
    
    # Ensure weighted_score exists
    if 'weighted_score' not in df.columns:
         df['weighted_score'] = df['top3_hit'] * 0.6 + df['map_match_rate'] * 0.2 + df['posterior_consistency'] * 0.2

    # --- Data Aggregation ---
    # Since we have 5 dimensions, we maximize over the others (delta, alpha, kappa, beta)
    # to find the "potential" of each (rho, gamma) pair.
    # This creates a "Roofline" surface of the optimization landscape.
    pivot_table = df.pivot_table(
        index='gamma', 
        columns='rho', 
        values='weighted_score', 
        aggfunc='max'
    )
    
    # Sort index for correct plotting orientation
    pivot_table = pivot_table.sort_index(ascending=False) # standard y-axis orientation

    # --- Plotting ---
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Create Heatmap
    # Using 'viridis' - the gold standard for scientific visualization (perceptually uniform)
    ax = sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".3f", 
        cmap="viridis", 
        linewidths=.5,
        cbar_kws={'label': 'Weighted Score'},
        annot_kws={'size': 9}
    )
    
    # --- Formatting ---
    plt.title('Hyperparameter Optimization Surface (Weighted Score)', fontsize=14, weight='bold', pad=20)
    plt.xlabel(r'Memory Decay Factor ($\rho$)', fontsize=12, labelpad=10)
    plt.ylabel(r'Judge Influence Factor ($\gamma$)', fontsize=12, labelpad=10)
    
    # Improve tick labels
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    # Save
    output_path = 'output/optimization_landscape.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_landscape()
