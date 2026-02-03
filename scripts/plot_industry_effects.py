import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def create_industry_dumbbell_chart():
    # Data from Table 17
    data = {
        'Industry': ['Politician', 'Journalist', 'Reality Star', 'Athlete', 'Musician', 'TV Personality', 'Model', 'Comedian', 'Actor'],
        'Judge_Coef': [-1.05, -1.11, 0.30, -0.17, 0.34, -0.37, -0.34, -0.22, 0.00],
        'Fan_Bias':   [0.13, -1.20, -1.01, 0.02, -0.34, -0.38, -0.73, -0.64, 0.00],
        'Difference': [1.18, -0.09, -1.31, 0.19, -0.68, -0.01, -0.39, -0.42, 0.00]
    }
    
    df = pd.DataFrame(data)
    
    # Sort by Judge Coef for better visual flow, or by Difference to highlight the gap?
    # Let's sort by 'Difference' to show the spectrum from "Fan Favorite" to "Judge Favorite".
    df = df.sort_values(by='Difference', ascending=False)
    
    # Setup Style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'sans-serif']
    sns.set_theme(style="whitegrid", context="talk")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors
    color_judge = '#c1de9c' # Pastel Green from previous charts
    color_fan = '#66bdce'   # Pastel Blue from previous charts
    color_line = '#aaaaaa'
    
    # Create Dumbbell Plot
    for i, row in df.iterrows():
        # Line connecting the dots
        ax.plot([row['Judge_Coef'], row['Fan_Bias']], [i, i], color=color_line, zorder=1, linewidth=2, alpha=0.6)
        
        # Difference Annotation in the middle of the line
        mid_point = (row['Judge_Coef'] + row['Fan_Bias']) / 2
        diff = row['Difference']
        
        # Add arrow marker to show direction if gap is large enough
        if abs(diff) > 0.1:
             # Just the text for now to keep it clean, maybe color code the text?
             # If Diff > 0 (Fan > Judge), color Blue. If Diff < 0 (Judge > Fan), color Green.
             text_color = color_fan if diff > 0 else color_judge
             font_weight = 'bold' if abs(diff) > 0.5 else 'normal'
             
             # Nudge text up slightly
             ax.text(mid_point, i + 0.15, f"Δ {diff:+.2f}", 
                     ha='center', va='center', fontsize=9, color=text_color, fontweight=font_weight)

    # Plot Scatter Points
    # Judge
    ax.scatter(df['Judge_Coef'], range(len(df)), color=color_judge, s=150, label='Judge Preference', zorder=3, edgecolors='white', linewidth=1.5)
    # Fan
    ax.scatter(df['Fan_Bias'], range(len(df)), color=color_fan, s=150, label='Fan Bias', zorder=3, edgecolors='white', linewidth=1.5)
    
    # Axis formatting
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Industry'], fontsize=12, fontweight='bold')
    
    # Add vertical line at 0 (Reference: Actor)
    ax.axvline(0, color='#cccccc', linestyle='--', linewidth=1, zorder=0)
    ax.text(0.02, -0.8, 'Reference (Actor)', fontsize=10, color='#999999')
    
    # Labels and Title
    ax.set_xlabel('Coefficient Value (Relative to Actor)', fontsize=11, labelpad=10)
    # ax.set_title('The Populist Gap: Divergence between Judge & Fan Preferences by Industry', fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, framealpha=0.9)
    
    # Annotate significant points
    # Politician
    pol_idx = df[df['Industry'] == 'Politician'].index[0]
    # ax.text(-1.1, pol_idx, "Judges disapprove\n(-1.05)", ha='left', va='center', fontsize=9, color=color_judge)
    
    plt.tight_layout()
    output_path = 'output/industry_bias_dumbbell.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    create_industry_dumbbell_chart()
