"""
Global Parameter Optimization for p_concave
Sweeps p from [0.1, 1.0] across ALL seasons to find the global optimum
for the Objective Function: Score = 0.3 * min(Fan, 0.75) + 0.7 * Judge
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.acdw_rule import ACDWRule
from core.competition_rules import CompetitionRules

# Helper for alignment (same as benchmark)
def calculate_alignment(active_names, judge_scores, vote_shares, eliminated_set):
    n = len(active_names)
    if n <= 1: return 1.0, 1.0
    
    from scipy.stats import rankdata
    f_ranks = rankdata([-vote_shares.get(n, 0) for n in active_names], method='average')
    f_map = {n: r for n, r in zip(active_names, f_ranks)}
    j_ranks = rankdata([-judge_scores.get(n, 0) for n in active_names], method='average')
    j_map = {n: r for n, r in zip(active_names, j_ranks)}
    
    if not eliminated_set: return 0.0, 0.0
    
    avg_f = np.mean([f_map.get(e, n) for e in eliminated_set])
    avg_j = np.mean([j_map.get(e, n) for e in eliminated_set])
    
    return (avg_f - 1)/(n - 1), (avg_j - 1)/(n - 1)

def optimize_p():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output', 'vote_share_estimates.json')
    
    dp = DataProcessor(csv_path)
    with open(json_path, 'r') as f:
        estimates = json.load(f)
        
    p_values = np.arange(0.1, 1.05, 0.05)
    results = []
    
    # Pre-load data to speed up loop
    simulation_contexts = []
    sorted_seasons = sorted([int(k) for k in estimates.keys()])
    
    print("Pre-loading simulation contexts...")
    for season in sorted_seasons:
        events = dp.get_elimination_events(season)
        real_elim = {e.week: set(e.eliminated_set) for e in events}
        weeks_data = estimates[str(season)]
        
        for week_str, est in weeks_data.items():
            week = int(week_str)
            if week not in real_elim: continue
            
            active = dp.get_active_contestants(season, week)
            active_names = [c.name for c in active]
            if len(active_names) < 3: continue 
            
            judge_scores = {c.name: dp.get_weekly_total_score(c, week) for c in active}
            
            # Reconstruct Shares
            shares = {}
            for name, stats in est.items():
                if name in active_names: shares[name] = stats['mean']
            total = sum(shares.values())
            if total == 0: continue
            p_shares = {k: v/total for k, v in shares.items()}
            
            simulation_contexts.append({
                'season': season, 'week': week,
                'active': active_names,
                'judge': judge_scores,
                'fan': p_shares,
                'n_elim': len(real_elim[week])
            })
            
    print(f"Loaded {len(simulation_contexts)} contexts. Starting Sweep...")
    
    for p in tqdm(p_values):
        # We assume lambda is dynamic, but fixed params for lambda function
        rule = ACDWRule(p_concave=p, lambda_min=0.6, lambda_max=0.8)
        
        f_scores = []
        j_scores = []
        
        for ctx in simulation_contexts:
            # Deterministic run for speed (Average behavior)
            # Alignment is robust enough without jitter for trend analysis
            outcome = rule.compute_outcome(ctx['judge'], ctx['fan'], n_eliminated=ctx['n_elim'])
            
            fa, ja = calculate_alignment(ctx['active'], ctx['judge'], ctx['fan'], outcome.eliminated)
            f_scores.append(fa)
            j_scores.append(ja)
            
        avg_f = np.mean(f_scores)
        avg_j = np.mean(j_scores)
        
        # Calculate Objective Score
        # w_F = 0.3 (Capped at 0.75), w_J = 0.7
        capped_f = min(avg_f, 0.75)
        score = 0.3 * capped_f + 0.7 * avg_j
        
        results.append({
            'p': p,
            'Fan_Align': avg_f,
            'Judge_Align': avg_j,
            'Total_Score': score
        })
        
    # Find Best
    df = pd.DataFrame(results)
    best_row = df.loc[df['Total_Score'].idxmax()]
    
    print("\nOptimization Results:")
    pd.set_option('display.max_rows', None)
    print(df[['p', 'Fan_Align', 'Judge_Align', 'Total_Score']])
    print(f"\nOptimal P: {best_row['p']:.2f} (Score: {best_row['Total_Score']:.4f})")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['p'], df['Fan_Align'], label='Fan Alignment', marker='o')
    plt.plot(df['p'], df['Judge_Align'], label='Judge Alignment', marker='s')
    plt.plot(df['p'], df['Total_Score'], label='Total Score', linewidth=3, color='black')
    
    plt.axvline(best_row['p'], color='red', linestyle='--', label=f'Optimal p={best_row["p"]:.2f}')
    
    plt.xlabel('p (Concavity)')
    plt.ylabel('Score')
    plt.title('Global Parameter Optimization: Finding the Optimal p')
    plt.legend()
    plt.grid(True)
    
    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images', 'p_optimization.png')
    plt.savefig(img_path)
    print(f"Plot saved to {img_path}")

if __name__ == "__main__":
    optimize_p()
