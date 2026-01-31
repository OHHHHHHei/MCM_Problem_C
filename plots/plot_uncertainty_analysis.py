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

def plot_uncertainty():
    # 1. Config
    csv_path = '2026_MCM_Problem_C_Data.csv'
    
    # Optimized Parameters
    best_params = ModelParams(
        n_particles=400,
        rho=0.9,
        gamma=0.0,
        delta=1.2,
        alpha_discriminant=7.0,
        kappa=0.3,
        beta_judge=2.0
    )
    
    print("Running model for Uncertainty Analysis...")
    
    dp = DataProcessor(csv_path)
    model = SMCInverse(dp, best_params)
    
    # 2. Run All Seasons
    seasons = [s for s in dp.get_seasons() if s != 32]
    results = model.run_all_seasons(seasons=seasons, verbose=False)
    
    # 3. Collect CI Width Data
    ci_data = []  # List of dicts: {season, week, name, ci_width, status, phase}
    
    print("Extracting CI width data...")
    
    for season, res in results.items():
        estimates = res.get('weekly_estimates', {})
        events = dp.get_elimination_events(season)
        elim_map = {e.week: set(e.eliminated_set) for e in events}
        
        # Get total weeks for phase calculation
        all_weeks = sorted(estimates.keys())
        n_weeks = len(all_weeks)
        
        for i, week in enumerate(all_weeks):
            # Use Absolute Week Number (1-indexed)
            week_num = i + 1
            
            week_est = estimates[week]
            actual_eliminated = elim_map.get(week, set())
            
            for name, est in week_est.items():
                ci_width = est.get('ci_width', est['ci_high'] - est['ci_low'])
                status = 'Eliminated' if name in actual_eliminated else 'Survivor'
                
                ci_data.append({
                    'season': season,
                    'week': week,
                    'week_num': week_num, # Absolute week number
                    'name': name,
                    'ci_width': ci_width,
                    'status': status
                })

    df = pd.DataFrame(ci_data)
    print(f"Collected {len(df)} data points.")
    
    # 4. Plotting - Two Panel Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # --- Panel 1: Boxplot by Status ---
    ax1 = axes[0]
    
    palette = {'Eliminated': '#d62728', 'Survivor': '#2ca02c'}
    
    sns.boxplot(
        data=df, 
        x='status', 
        y='ci_width', 
        palette=palette,
        order=['Eliminated', 'Survivor'],
        ax=ax1,
        width=0.5
    )
    
    # Add statistical annotation
    elim_mean = df[df['status'] == 'Eliminated']['ci_width'].mean()
    surv_mean = df[df['status'] == 'Survivor']['ci_width'].mean()
    ratio = surv_mean / elim_mean
    
    ax1.set_xlabel('Contestant Status (in each week)', fontsize=12)
    ax1.set_ylabel('95% Credible Interval Width', fontsize=12)
    ax1.set_title('(a) Uncertainty by Status:\nEliminated Contestants are More Predictable', fontsize=13, weight='bold')
    
    # Add text annotation inside plot
    ax1.text(0.5, 0.95, f'Ratio (Survivor/Eliminated): {ratio:.2f}x',
             transform=ax1.transAxes, ha='center', va='top', fontsize=11,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))
    
    # --- Panel 2: Line Plot by Absolute Week ---
    ax2 = axes[1]
    
    # Filter for first 10 weeks (common to most seasons) to avoid noise from long seasons
    max_week = 10
    df_week = df[df['week_num'] <= max_week]
    
    # Aggregate by week number
    week_stats = df_week.groupby('week_num')['ci_width'].agg(['mean', 'std'])
    
    x_weeks = week_stats.index
    
    ax2.errorbar(
        x_weeks, 
        week_stats['mean'], 
        yerr=week_stats['std'], 
        marker='o', 
        markersize=8, 
        linewidth=2.5, 
        capsize=5, 
        capthick=2,
        color='#1f77b4',
        label='Mean CI Width'
    )
    
    ax2.set_xlabel('Competition Week (Absolute)', fontsize=12)
    ax2.set_ylabel('95% Credible Interval Width (Mean ± Std)', fontsize=12)
    ax2.set_title('(b) Uncertainty Evolution:\nRapid Convergence in Early Weeks', fontsize=13, weight='bold')
    ax2.set_xticks(x_weeks)
    
    # Add trend interpretation
    first_val = week_stats['mean'].iloc[0]
    min_val = week_stats['mean'].min()
    drop_pct = (first_val - min_val) / first_val
    
    ax2.annotate(f'-{drop_pct:.0%} Uncertainty\n(Learning Effect)', 
                 xy=(5, week_stats.loc[5, 'mean']), 
                 xytext=(5, first_val),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1),
                 fontsize=10, ha='center')
    
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # Overall title
    fig.suptitle('Uncertainty Heterogeneity Analysis: Credible Interval Width is NOT Constant', 
                 fontsize=15, weight='bold', y=1.02)
    
    # Save
    output_path = 'output/uncertainty_heterogeneity.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_uncertainty()
