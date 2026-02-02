"""
Parameter Sweep Visualization: Finding the Sweet Spot for p_concave
Focus: S27 Bobby Bones (Weeks 6-8)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.acdw_rule import ACDWRule

def plot_parameter_sweep():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    dp = DataProcessor(csv_path)
    
    # Target: Bobby Bones, S27, Week 6 (Example of high controversy)
    target_contestant = "Bobby Bones"
    season = 27
    week = 6  # A week where he performed poorly but had high votes
    
    print(f"Analyzing {target_contestant} in Season {season}, Week {week}...")
    
    # Get Data
    active = dp.get_active_contestants(season, week)
    active_names = [c.name for c in active]
    
    # Real Judge Scores
    judge_scores = {c.name: dp.get_weekly_total_score(c, week) for c in active}
    
    # Estimated Vote Shares (Approximation from Q2 or hardcoded scenario)
    # Scenario: Bobby ~40%, Others split remaining
    # Let's use a realistic distribution based on Q2 analysis
    # Bobby: 0.35, Others: ~0.08
    # Normalize to be sure
    remaining_share = 0.65
    n_others = len(active_names) - 1
    vote_shares = {}
    for name in active_names:
        if name == target_contestant:
            vote_shares[name] = 0.35
        else:
            vote_shares[name] = remaining_share / n_others
            
    # Sweep p
    p_values = np.linspace(0.1, 1.0, 50)
    bobby_ranks = []
    bobby_scores = []
    safety_margins = [] # Score difference from "Cutoff" (Bottom 1 or 2)
    
    for p in p_values:
        # Instantiate Rule
        rule = ACDWRule(p_concave=p, lambda_min=0.6, lambda_max=0.8, protection_level=0)
        
        # Compute Outcome
        # We need the raw scores to see margin.
        # acdw.compute_outcome returns result with scores
        res = rule.compute_outcome(judge_scores, vote_shares, n_eliminated=1)
        
        scores = res.survival_scores
        
        # Bobby's Score
        b_score = scores[target_contestant]
        
        # Rank (1 = Lowest Score = Eliminated)
        # sorted_scores: Low to High
        sorted_s = sorted(scores.items(), key=lambda x: x[1])
        # Find Bobby's index
        rank = -1
        for i, (name, s) in enumerate(sorted_s):
            if name == target_contestant:
                rank = i + 1 # 1-based, 1 is Worst
                break
                
        # Margin: Distance from "Safe Zone"
        # If Rank 1 (Worst), Margin is (MyScore - NextScore) (Negative)
        # If Safe, Margin is (MyScore - WorstScore) (Positive)
        worst_score = sorted_s[0][1]
        second_worst = sorted_s[1][1]
        
        if rank == 1:
            margin = b_score - second_worst # Negative
        else:
            margin = b_score - worst_score # Positive
            
        bobby_ranks.append(rank)
        safety_margins.append(margin)
        
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot Margin
    plt.plot(p_values, safety_margins, label='Safety Margin (Score Diff)', color='blue', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', label='Elimination Threshold')
    
    # Identify p=0.55
    plt.axvline(0.55, color='green', linestyle=':', label='Chosen p=0.55')
    
    plt.title(f"Parameter Sweep: Impact of p on {target_contestant} (S{season} W{week})")
    plt.xlabel('Concavity Parameter (p)')
    plt.ylabel('Safety Margin (Positive = Safe, Negative = Eliminated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images', 'parameter_sweep_bobby.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    
    # Print range analysis
    print(f"P-Value Range: {p_values[0]:.2f} to {p_values[-1]:.2f}")
    print(f"Min Margin: {min(safety_margins):.4f}")
    print(f"Max Margin: {max(safety_margins):.4f}")
    
    # Explicitly check for crossing
    crossing_found = False
    for i in range(len(safety_margins)-1):
        if (safety_margins[i] < 0 and safety_margins[i+1] > 0) or (safety_margins[i] > 0 and safety_margins[i+1] < 0):
            print(f"CROSSING FOUND between p={p_values[i]:.2f} (M={safety_margins[i]:.4f}) and p={p_values[i+1]:.2f} (M={safety_margins[i+1]:.4f})")
            crossing_found = True
            
    if not crossing_found:
        print("NO CROSSING FOUND. Bobby is always " + ("SAFE" if safety_margins[0] > 0 else "ELIMINATED"))
        # Print a few sample points
        indices = [0, 10, 25, 40, 49] # Sample indices
        for i in indices:
            if i < len(p_values):
                print(f"p={p_values[i]:.2f}: Rank={bobby_ranks[i]}, Margin={safety_margins[i]:.4f}")

if __name__ == "__main__":
    plot_parameter_sweep()
