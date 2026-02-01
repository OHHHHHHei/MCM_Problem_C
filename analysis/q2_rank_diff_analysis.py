import pandas as pd
import numpy as np
import json
import os
import sys

# Add core to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.competition_rules import CompetitionRules, SurvivalScore
from core.data_processor import DataProcessor

def calculate_ranks(scores, reverse=False):
    """
    Returns a dictionary of ranks {name: rank}.
    If reverse=True, higher score = rank 1.
    If reverse=False, lower score = rank 1.
    """
    if isinstance(scores, dict):
        items = list(scores.items())
    else:
        return {}
    
    # Sort
    sorted_items = sorted(items, key=lambda x: x[1], reverse=reverse)
    
    ranks = {}
    for i, (name, _) in enumerate(sorted_items):
        ranks[name] = i + 1
    return ranks

def compute_controversy_scores(dp, vote_shares):
    """
    Compute S1 and S2 scores for all contestants.
    S1: Populist Score
    S2: Robbed Score
    """
    contestant_stats = []
    
    seasons = dp.get_seasons()
    for season in seasons:
        contestants = dp.get_contestants_in_season(season)
        N = len(contestants)
        if N <= 1: continue
        
        # Pre-calculate judge rankings per week
        # And getting Final Placement
        # And getting Z-scores
        
        # Get weekly Judge Ranks
        # We need average judge rank percentage
        # Iterate all weeks in season
        # But we need weeks where they were active
        
        contestant_map = {c.name: c for c in contestants}
        # Get max week
        max_week = max([max(c.weekly_scores.keys()) for c in contestants if c.weekly_scores] or [0])
        
        # Store judge ranks per contestant per week
        judge_ranks_history = {c.name: [] for c in contestants}
        z_scores_history = {c.name: [] for c in contestants}
        
        for week in range(1, max_week + 1):
            active = dp.get_active_contestants(season, week)
            if not active: continue
            
            # Compute Judge Scores
            j_scores = {c.name: dp.get_weekly_average_score(c, week) for c in active}
            if not j_scores: continue
            
            # Compute Ranks (Higher Score = Better Rank = Lower Number)
            sorted_j = sorted(j_scores.items(), key=lambda x: x[1], reverse=True)
            ranks = {name: i+1 for i, (name, s) in enumerate(sorted_j)}
            
            # Compute Z-Scores
            vals = list(j_scores.values())
            mu = np.mean(vals)
            sigma = np.std(vals) if len(vals) > 1 else 1.0
            if sigma < 0.001: sigma = 1.0
            
            for name, score in j_scores.items():
                judge_ranks_history[name].append(ranks[name])
                z = (score - mu) / sigma
                z_scores_history[name].append(z)
                
        # Now compute S1 and S2 for each contestant
        for c in contestants:
            name = c.name
            
            # --- S1 Computation ---
            # R_bar_pct
            ranks = judge_ranks_history.get(name, [])
            if not ranks:
                s1 = 0
            else:
                avg_rank = np.mean(ranks)
                std_rank = np.std(ranks) if len(ranks) > 1 else 0
                
                # Normalize
                r_bar_pct = (avg_rank - 1) / (N - 1)
                
                # P_pct
                p_final = c.placement if pd.notna(c.placement) else N
                p_pct = (p_final - 1) / (N - 1)
                
                # Sigma_norm
                sigma_norm = std_rank / (N - 1)
                
                # Formula: max(0, r_bar_pct - p_pct) * (1 - sigma_norm)
                term1 = max(0, r_bar_pct - p_pct)
                s1 = term1 * (1 - sigma_norm)
            
            # --- S2 Computation ---
            # Z_last, Z_bar
            z_hist = z_scores_history.get(name, [])
            if not z_hist:
                s2 = 0
            else:
                # Last week is the last entry in history
                z_last = z_hist[-1]
                z_mean = np.mean(z_hist)
                
                # Formula: max(0, z_last) * max(0.5, 1 + z_mean)
                # Special rule: Winner (placement=1) -> S2=0
                if c.placement == 1:
                    s2 = 0
                else:
                    s2 = max(0, z_last) * max(0.5, 1 + z_mean)
            
            contestant_stats.append({
                'Season': season,
                'Contestant': name,
                'S1': s1,
                'S2': s2
            })
            
    return pd.DataFrame(contestant_stats)

def analyze_rank_diff_for_subsets():
    print("Loading data...")
    # Load Data Processor
    csv_path = 'data/raw/2026_MCM_Problem_C_Data.csv'
    if not os.path.exists(csv_path):
        csv_path = '2026_MCM_Problem_C_Data.csv'
    dp = DataProcessor(csv_path)
    
    # Load Vote Shares
    with open('output/vote_share_estimates.json', 'r') as f:
        vote_shares = json.load(f)
        
    print("Computing Controversy Scores (S1, S2)...")
    df_scores = compute_controversy_scores(dp, vote_shares)
    df_scores.to_csv('output/controversy_scores.csv', index=False)
    print("Saved controversy scores to output/controversy_scores.csv")
    
    # Identify Top 10% for S1 and S2
    # Thresholds
    n_total = len(df_scores)
    n_top = int(n_total * 0.1)
    
    top_s1 = df_scores.sort_values('S1', ascending=False).head(n_top)
    top_s2 = df_scores.sort_values('S2', ascending=False).head(n_top)
    
    targets_s1 = set(zip(top_s1['Season'], top_s1['Contestant']))
    targets_s2 = set(zip(top_s2['Season'], top_s2['Contestant']))
    
    print(f" identified Top {n_top} contestants for S1 and S2 metrics.")
    
    # Now analyze Rank Diff for these targets
    results_s1 = []
    results_s2 = []
    
    # Iterate all seasons
    seasons = dp.get_seasons() # All seasons S1-S34
    
    for season in seasons:
        # Check if we have estimates
        estimates = vote_shares.get(str(season), {})
        if not estimates: continue
        
        weeks = sorted([int(w) for w in estimates.keys()])
        
        for week in weeks:
            active_objs = dp.get_active_contestants(season, week)
            active_names = [c.name for c in active_objs]
            # Intersection with estimates
            week_est = estimates[str(week)]
            active_names = [n for n in active_names if n in week_est]
            
            if not active_names: continue
            
            # --- Rule Simulations (Same logic as before) ---
            # 1. Judge Scores
            j_scores_map = {}
            for c in active_objs:
                if c.name in active_names:
                    j_scores_map[c.name] = dp.get_weekly_total_score(c, week)
                    
            # 2. Vote Shares
            v_shares_map = {n: week_est[n]['mean'] for n in active_names}
            
            # --- Rank Rule ---
            # Rank J (High score = Rank 1)
            sorted_j = sorted(j_scores_map.items(), key=lambda x: x[1], reverse=True)
            rank_j_map = {n: i+1 for i, (n, _) in enumerate(sorted_j)}
            
            # Rank V (High share = Rank 1)
            sorted_v = sorted(v_shares_map.items(), key=lambda x: x[1], reverse=True)
            rank_v_map = {n: i+1 for i, (n, _) in enumerate(sorted_v)}
            
            # Total Rank Score: -(Rj + Rv) -> Higher is better
            rank_rule_scores = {n: -(rank_j_map[n] + rank_v_map[n]) for n in active_names}
            final_ranks_rank = calculate_ranks(rank_rule_scores, reverse=True)
            
            # --- Pct Rule ---
            total_j = sum(j_scores_map.values()) if j_scores_map else 1
            pct_scores = {n: (j_scores_map[n]/total_j) + v_shares_map[n] for n in active_names}
            final_ranks_pct = calculate_ranks(pct_scores, reverse=True)
            
            # --- Check Targets ---
            def check_target(targets, res_list, subset_name):
                for name in active_names:
                    if (season, name) in targets:
                        r_rank = final_ranks_rank[name]
                        r_pct = final_ranks_pct[name]
                        diff = r_pct - r_rank
                        
                        # Judge Save Logic: In Bottom 2?
                        n_active = len(active_names)
                        in_b2_rank = r_rank >= (n_active - 1)
                        # Did Save change fate?
                        # This requires simulating Save outcome.
                        # Simple proxy: Is in Bottom 2 ONLY in Rank Rule? Or ONLY in Pct?
                        # Or does Save rescue them?
                        
                        res_list.append({
                            'Season': season,
                            'Week': week,
                            'Contestant': name,
                            'Subset': subset_name,
                            'Rank_RankRule': r_rank,
                            'Rank_PctRule': r_pct,
                            'Rank_Diff': diff,
                            'In_B2_Rank': in_b2_rank
                        })

            check_target(targets_s1, results_s1, 'S1_High_Populist')
            check_target(targets_s2, results_s2, 'S2_High_Robbed')

    def print_summary(res_list, name):
        if not res_list:
            print(f"\nNo data for {name}")
            return
        df = pd.DataFrame(res_list)
        print(f"\n=== Summary for {name} (Top 10%) ===")
        print(f"Total Observations (Contestant-Weeks): {len(df)}")
        print(f"Avg Rank Diff (Pct_Rank - Rank_Rank): {df['Rank_Diff'].mean():.2f}")
        print("Positive Diff means Rank Rule gave BETTER placement (Lower Rank Num) than Pct Rule.")
        print("(i.e., Contestant does WORSE in Pct Rule)")
        
        # S1 (Populist): We expect them to do BETTER in Pct (Rank_Diff < 0) or Worse in Rank?
        # Pct Rule allows votes to swamp judges. Rank rule caps votes.
        # So Populists should have Better Rank (Lower Num) in Pct.
        # So Rank_Pct < Rank_Rank => Diff < 0.
        
        print(f"Avg Diff: {df['Rank_Diff'].mean():.2f}")
        print(f"Proportion where Pct Rank is Better (Diff < 0): {(df['Rank_Diff'] < 0).mean():.2%}")
        print(f"Proportion where Pct Rank is Worse (Diff > 0): {(df['Rank_Diff'] > 0).mean():.2%}")
        
        # Judge Save Impact Potential
        # How often in Bottom 2 under Rank Rule?
        b2_rate = df['In_B2_Rank'].mean()
        print(f"Frequency in Bottom 2 (Rank Rule): {b2_rate:.2%}")
        
        # Detailed stats about Diff magnitude
        print(f"Max Benefit from Pct Rule (Min Diff): {df['Rank_Diff'].min()}")
        print(f"Max Penalty from Pct Rule (Max Diff): {df['Rank_Diff'].max()}")

    print_summary(results_s1, "S1 (Populist / High Vote Low Judge)")
    print_summary(results_s2, "S2 (Robbed / High Judge Low Result)")
    
    # Save CSVs
    pd.DataFrame(results_s1).to_csv('output/q2_S1_analysis.csv', index=False)
    pd.DataFrame(results_s2).to_csv('output/q2_S2_analysis.csv', index=False)

if __name__ == "__main__":
    analyze_rank_diff_for_subsets()
