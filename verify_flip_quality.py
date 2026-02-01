
import json
import numpy as np
import sys
import os
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.getcwd())

from core.data_processor import DataProcessor
from core.competition_rules import CompetitionRules
from core.smc_inverse import ParticleState
from core.counterfactual import CounterfactualSimulator

def verify_flip_quality():
    print("Loading data for Flip Quality Verification...")
    dp = DataProcessor("2026_MCM_Problem_C_Data.csv")
    
    with open("output/vote_share_estimates.json", "r") as f:
        estimates_data = json.load(f)
        
    rules = CompetitionRules()
    simulator = CounterfactualSimulator(rules, n_particles=50) # Faster run for verification
    
    diff_scores = []
    
    print("Simulating flips...")
    
    for season_str, weeks_data in tqdm(estimates_data.items()):
        season = int(season_str)
        
        # Get actual elimination amount
        events = dp.get_elimination_events(season)
        real_eliminated_by_week = {e.week: set(e.eliminated_set) for e in events}
        
        for week_str, est_dict in weeks_data.items():
            week = int(week_str)
            if week not in real_eliminated_by_week: continue
                
            n_eliminated = len(real_eliminated_by_week[week])
            active = dp.get_active_contestants(season, week)
            active_names = [c.name for c in active]
            # Normalized Judge Scores for fair comparison across weeks
            raw_judge_scores = {c.name: dp.get_weekly_total_score(c, week) for c in active}
            max_score = max(raw_judge_scores.values()) if raw_judge_scores else 1
            judge_scores = {k: v/max_score for k, v in raw_judge_scores.items()}
            
            # Construct Particle
            mean_shares = {}
            for name, stats in est_dict.items():
                if name in active_names:
                    mean_shares[name] = stats['mean']
            
            total_share = sum(mean_shares.values())
            if total_share == 0: continue
            mean_shares = {k: v/total_share for k, v in mean_shares.items()}
            x_values = {name: np.log(share + 1e-10) for name, share in mean_shares.items()}
            
            dummy_particle = ParticleState(
                mu={}, x=x_values, weight=1.0,
                accumulated_shares={n: 0.0 for n in active_names},
                accumulated_vote_ranks={n: 0.0 for n in active_names}
            )
            particles = [dummy_particle] * 50
            
            # Run Rank vs Pct
            res_rank = simulator.simulate_single_week(
                season, week, particles, active_names, raw_judge_scores, # Pass raw, let rules handle it
                rule_type='RANK', n_eliminated=n_eliminated
            )
            res_pct = simulator.simulate_single_week(
                season, week, particles, active_names, raw_judge_scores,
                rule_type='PERCENTAGE', n_eliminated=n_eliminated
            )
            
            # Analyze Flips
            for r_set, p_set in zip(res_rank, res_pct):
                if r_set != p_set:
                    # Rank Eliminated these
                    # Pct Eliminated these
                    # We want to know: Did Rank eliminate WORSE dancers?
                    
                    # Avg Judge Score of Rank Eliminated
                    score_rank_elim = np.mean([judge_scores.get(n, 0) for n in r_set])
                    
                    # Avg Judge Score of Pct Eliminated
                    score_pct_elim = np.mean([judge_scores.get(n, 0) for n in p_set])
                    
                    # Delta = Rank_Elim_Score - Pct_Elim_Score
                    # If Delta < 0, it means Rank eliminated people with LOWER scores (Better Meritocracy)
                    diff_scores.append(score_rank_elim - score_pct_elim)

    avg_diff = np.mean(diff_scores)
    
    print("\n" + "="*50)
    print("FLIP QUALITY ANALYSIS")
    print("="*50)
    print(f"Total Flips Analyzed: {len(diff_scores)}")
    print(f"Average Judge Score Diff (RankElim - PctElim): {avg_diff:.4f}")
    
    if avg_diff < 0:
        print("CONCLUSION: Rank Rule consistently eliminates contestants with LOWER judge scores.")
        print("            (Verified: Rank Rule protects talent better)")
    else:
        print("CONCLUSION: Rank Rule eliminates contestants with HIGHER judge scores.")
        print("            (Hypothesis FAILED)")

if __name__ == "__main__":
    verify_flip_quality()
