import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def plot_goldilocks():
    # Adjust path to output file since we are in plots/
    input_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_overnight_results.csv'))
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print("Loading data...")
    df = pd.read_csv(input_file)
    
    if 'weighted_score' not in df.columns:
         df['weighted_score'] = df['top3_hit'] * 0.6 + df['map_match_rate'] * 0.2 + df['posterior_consistency'] * 0.2

    # --- Data Filtering Strategy ---
    # We want to show the "Potential" of each rho. 
    # If we just average everything, bad settings of gamma/alpha will drown out the signal.
    # So, for each rho, we take the TOP 50 results (representing a "well-tuned" model).
    
    top_n = 50
    filtered_df = df.groupby('rho').apply(
        lambda x: x.nlargest(top_n, 'weighted_score')
    ).reset_index(drop=True)
    
    print("\nStats per Rho (Top 50 samples):")
    stats = filtered_df.groupby('rho')['weighted_score'].agg(['mean', 'max', 'std'])
    print(stats)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # Main Error Bar Plot
    # seaborn lineplot automatically calculates mean and confidence interval/std dev
    sns.lineplot(
        data=filtered_df, 
        x='rho', 
        y='weighted_score', 
        marker='o', 
        markersize=8,
        linewidth=2.5,
        errorbar='sd', # Show Standard Deviation as error bars
        color='#2ca02c', # Muted Green
        ax=ax
    )

    # Formatting
    ax.set_xlabel(r'Memory Decay Factor ($\rho$)', fontsize=14, labelpad=10)
    ax.set_ylabel('Weighted Score (Top-50 Avg)', fontsize=14, labelpad=10)
    
    plt.title('The "Memory Effect":\nPerformance Increases with Fan Memory Retention', fontsize=16, weight='bold', pad=20)
    
    # Annotate the peak
    # Find the rho with the highest mean score in our filtered set
    best_rho = stats['mean'].idxmax()
    best_mean = stats['mean'].max()
    
    ax.annotate(f'Optimal Point: $\\rho={best_rho}$\n(Long-term Memory)', 
                 xy=(best_rho, best_mean), 
                 xytext=(best_rho - 0.15, best_mean - 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                 fontsize=11, weight='bold')

    # Add text explaining the trend
    story_text = (
        "Trend: Higher $\\rho$ correlates with better accuracy.\n"
        "Interpretation: Fans have long memories;\n"
        "accumulated popularity dominates momentary performance."
    )
    plt.figtext(0.15, 0.2, story_text, fontsize=11, 
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', edgecolor='gray'))

    ax.grid(True, linestyle=':', alpha=0.6)

    # Save
    output_path = 'output/memory_goldilocks_analysis.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_goldilocks()
