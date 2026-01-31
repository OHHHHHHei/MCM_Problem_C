import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from collections import defaultdict

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

# Custom Pastel Color Palette
COLORS = {
    'apricot': '#d6e0ed',   # 浅杏色
    'lavender': '#b8b9d2',  # 浅紫色
    'mint': '#b5d4be',      # 浅绿色
    'sky': '#66bdce',       # 浅蓝色
}

def plot_uncertainty(force_rerun=False):
    # 1. Config
    csv_path = '2026_MCM_Problem_C_Data.csv'
    cache_file = 'output/uncertainty_data.csv'
    
    # Check if cache exists
    if os.path.exists(cache_file) and not force_rerun:
        print(f"Loading cached uncertainty data from {cache_file}...")
        df = pd.read_csv(cache_file)
        print(f"Loaded {len(df)} data points.")
    else:
        print("Cache not found or rerun requested. Running full simulation (1000 particles)...")
        
        # Optimized Parameters
        best_params = ModelParams(
            n_particles=1000,  # User requested 1000 particles for high precision
            rho=0.9,
            gamma=0.0,
            delta=1.2,
            alpha_discriminant=7.0,
            kappa=0.3,
            beta_judge=2.0
        )
        
        dp = DataProcessor(csv_path)
        model = SMCInverse(dp, best_params)
        
        # Run All Seasons
        seasons = [s for s in dp.get_seasons() if s != 32]
        results = model.run_all_seasons(seasons=seasons, verbose=False)
        
        # Collect CI Width Data
        ci_data = []
        print("Extracting CI width data...")
        
        for season, res in results.items():
            estimates = res.get('weekly_estimates', {})
            events = dp.get_elimination_events(season)
            elim_map = {e.week: set(e.eliminated_set) for e in events}
            
            all_weeks = sorted(estimates.keys())
            
            for i, week in enumerate(all_weeks):
                week_num = i + 1
                week_est = estimates[week]
                actual_eliminated = elim_map.get(week, set())
                
                for name, est in week_est.items():
                    ci_width = est.get('ci_width', est['ci_high'] - est['ci_low'])
                    status = 'Eliminated' if name in actual_eliminated else 'Survivor'
                    
                    ci_data.append({
                        'season': season,
                        'week': week,
                        'week_num': week_num,
                        'name': name,
                        'ci_width': ci_width,
                        'status': status
                    })
        
        df = pd.DataFrame(ci_data)
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        
        # Save to Cache
        df.to_csv(cache_file, index=False)
        print(f"Saved {len(df)} data points to {cache_file}")

    # 4. Plotting - Two Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # --- Panel 1: Beautiful Boxplot by Status ---
    ax1 = axes[0]
    
    # Use Pastel Colors: Eliminated=Lavender, Survivor=Mint
    palette = {'Eliminated': COLORS['lavender'], 'Survivor': COLORS['mint']}
    
    # Create prettier boxplot with stripplot overlay
    sns.boxplot(
        data=df, 
        x='status', 
        y='ci_width', 
        palette=palette,
        order=['Eliminated', 'Survivor'],
        ax=ax1,
        width=0.6,
        linewidth=1.5,
        fliersize=0,  # Hide outliers (will show as stripplot)
        boxprops=dict(edgecolor='#555555'),
        whiskerprops=dict(color='#555555'),
        capprops=dict(color='#555555'),
        medianprops=dict(color='#333333', linewidth=2)
    )
    
    # Add stripplot for individual points (jittered)
    sns.stripplot(
        data=df.sample(frac=0.1, random_state=42),  # Sample 10% for clarity
        x='status',
        y='ci_width',
        order=['Eliminated', 'Survivor'],
        ax=ax1,
        color='#666666',
        alpha=0.3,
        size=3,
        jitter=0.2
    )
    
    ax1.set_xlabel('Contestant Status (in each week)', fontsize=12)
    ax1.set_ylabel('Vote Share (95% CI Width)', fontsize=12)
    ax1.set_title('(a) Uncertainty Distribution by Contestant Status', fontsize=13, weight='bold')
    
    # Remove top and right spines for cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # --- Panel 2: Shaded Area Plot by Absolute Week ---
    ax2 = axes[1]
    
    # Filter for first 10 weeks (common to most seasons) to avoid noise from long seasons
    max_week = 10
    df_week = df[df['week_num'] <= max_week]
    
    # Aggregate by week number
    week_stats = df_week.groupby('week_num')['ci_width'].agg(['mean', 'std'])
    
    x_weeks = week_stats.index
    
    # Draw only mean line (no error bars)
    ax2.plot(
        x_weeks, 
        week_stats['mean'], 
        marker='o', 
        markersize=8, 
        linewidth=2.5, 
        color=COLORS['sky'],
        markeredgecolor=COLORS['sky'], # Match line color to hide edge effectively or set width 0
        markeredgewidth=0
    )
    
    # Fill between for confidence band
    ax2.fill_between(
        x_weeks,
        week_stats['mean'] - week_stats['std'],
        week_stats['mean'] + week_stats['std'],
        alpha=0.35,
        color=COLORS['apricot']
    )
    
    ax2.set_xlabel('Competition Week (Absolute)', fontsize=12)
    ax2.set_ylabel('Vote Share (95% CI Width)', fontsize=12)
    ax2.set_title('(b) Uncertainty Evolution Over Competition Weeks', fontsize=13, weight='bold')
    ax2.set_xticks(x_weeks)
    
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Overall title
    fig.suptitle('Uncertainty Heterogeneity Analysis', 
                 fontsize=15, weight='bold', y=1.02)
    
    # Save
    output_path = 'output/uncertainty_heterogeneity.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    # Set to True to force a fresh run (e.g., first time or params changed)
    # The user specifically asked to run it once now.
    # We default checking cache, but since we just wrote this, we want to run it.
    # However, to be safe, let's check args or just run it if file missing.
    # The user intent is "run simulation 1000 particles ONCE, save it, then use cache".
    
    import sys
    force = False
    if len(sys.argv) > 1 and sys.argv[1] == '--force':
        force = True
        
    # If cache doesn't exist, it will run automatically. 
    # To Ensure 1000 particles run happens now, user should probably delete old cache or we just rely on logic.
    # I'll add a flag logic to be robust.
    
    plot_uncertainty(force_rerun=force)
