import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

def create_industry_balloon_plot():
    # Data
    data = {
        # Sorted by Diff for better visual flow? Or keep original? Original is fine.
        'Industry': ['Politician', 'Journalist', 'Reality Star', 'Athlete', 'Musician', 'TV Personality', 'Model', 'Comedian', 'Actor'],
        'Judge Preference': [-1.05, -1.11, 0.30, -0.17, 0.34, -0.37, -0.34, -0.22, 0.00],
        'Fan Bias':   [0.13, -1.20, -1.01, 0.02, -0.34, -0.38, -0.73, -0.64, 0.00]
    }
    
    # Transform to long format
    df = pd.DataFrame(data)
    df_long = df.melt(id_vars='Industry', var_name='Metric', value_name='Value')
    
    # Pre-calculate lists for plotting logic
    industries = df['Industry'].tolist()
    metrics = ['Judge Preference', 'Fan Bias'] 
    
    # Setup styling
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
    sns.set_theme(style="white", context="talk", font='Times New Roman')
    
    # Use Equal Aspect Ratio for Square Boxes
    fig, ax = plt.subplots(figsize=(8.6, 3.4)) 
    ax.set_aspect('equal')
    
    # Grid - Disable automatic grid
    ax.grid(False)
    
    # Custom Grid: Draw individual box for each cell
    # Tiled appearance with NO gaps (size=1.0)
    # Figsize reduced to maintain physical box area
    for i in range(len(industries)):
        for j in [0, 1]:
            # Centered at (i, j), so start at i-0.5
            rect = plt.Rectangle((i-0.5, j-0.5), 1.0, 1.0, 
                               fill=False, edgecolor='#cccccc', linewidth=1)
            ax.add_patch(rect)

    # Custom Colormap
    # Colors: #f98c71 (Salmon), #fedf81 (Pale Orange), #eee683 (Yellow), #d3df82 (Green-Yellow)
    colors_list = ['#f98c71', '#fedf81', '#eee683', '#d3df82']
    custom_cmap = LinearSegmentedColormap.from_list('custom_gradient', colors_list)

    # Mapping
    # (industries and metrics defined above)
    x_vals, y_vals, sizes, colors = [], [], [], []
    
    for i, ind in enumerate(industries):
        for j, met in enumerate(metrics):
            val = df[df['Industry'] == ind][met].values[0]
            x_vals.append(i)
            # 1 for Judge (Top), 0 for Fan (Bottom)
            y_vals.append(1-j) 
            
            # Size mapping
            sizes.append(abs(val) * 600 + 150) 
            colors.append(val)

    # Plot
    sc = ax.scatter(x_vals, y_vals, s=sizes, c=colors, cmap=custom_cmap, 
                    vmin=-1.2, vmax=1.2, 
                    edgecolors='none', alpha=1.0)
    
    # Axis Config
    ax.set_xlim(-0.5, len(industries)-0.5)
    ax.set_ylim(-0.5, 1.5)
    
    # X Ticks (Top) - Vertical Labels
    ax.xaxis.tick_top()
    ax.set_xticks(range(len(industries)))
    ax.set_xticklabels(industries, rotation=90, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Y Ticks (Right/Left)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Fan Bias', 'Judge Preference'], fontsize=12, fontweight='bold')
    
    # Remove Spines entirely (Outer Frame Removed)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Hide Tick Marks but keep Labels
    ax.tick_params(length=0)
    
    # --- Custom Layout & Colorbar ---
    # Move plot down significantly (top=0.4) to accommodate tall vertical text
    plt.subplots_adjust(bottom=0.15, top=0.45, left=0.1, right=0.85)
    
    # We need to draw the canvas once to get the exact final position of the axes
    fig.canvas.draw() 
    pos = ax.get_position()
    
    # Create Custom Colorbar Axes (CAX)
    cbar_width = 0.02
    cbar_height = pos.height      # Align top and bottom perfectly with the grid
    cbar_bottom = pos.y0            
    cbar_left = pos.x1 + 0.02       
    
    cax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    
    cbar = plt.colorbar(sc, cax=cax)
    # cbar.set_label('Coefficient Value', rotation=270, labelpad=15) # Removed as requested
    
    # Match styles with the main grid boxes (Thin Line) but Black color as requested
    cbar.outline.set_linewidth(1)
    cbar.outline.set_edgecolor('black')
    
    # Save
    output_path = 'output/industry_bias_balloon.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Chart saved to {output_path}")

if __name__ == "__main__":
    create_industry_balloon_plot()
