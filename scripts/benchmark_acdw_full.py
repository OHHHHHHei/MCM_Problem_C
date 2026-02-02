
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse
from core.acdw_rule import ACDWRule

# --- Alignment Score Helper ---
# Recalculated from first principles to match Q2 definition
# Align_Fan(R) = P(Rule R saves Fan Favorite)
#   Fan Favorite = Candidate with Max Vote Share
#   Saved = Not Eliminated
# Align_Judge(R) = P(Rule R saves Judge Favorite)
#   Judge Favorite = Candidate with Max Judge Score

def calculate_alignment(active_names: List[str],
                        judge_scores: Dict[str, float],
                        vote_shares: Dict[str, float],
                        eliminated_set: Set[str]) -> Dict[str, float]:
    """
    Alignment Score: How well does the elimination align with preferences?
    Formula: (E[Rank_of_Eliminated] - 1) / (N - 1)
    
    Rank Definition: 
      - Rank 1 = Worst (Lowest Score/Share)
      - Rank N = Best (Highest Score/Share)
      - Ideally, we eliminate Rank 1 candidates.
      - So if Rank(Elim) = 1, Metric = (1-1)/(N-1) = 0? 
      - WAIT. User Formula says: "(Rank - 1) / (N - 1)"
      - If we eliminate the WORST person (Rank 1? or Rank N?), we want high alignment.
      - User's Q2 doc likely defined Rank 1 = Best? Let's check logic.
      - "Align_Fan = (E[Rank_pi(Elim)] - 1) / (N - 1)"
      - If Rank 1 is BEST (Highest Vote), then eliminating Rank 1 gives (1-1)/(N-1) = 0.
      - Eliminating Rank N (Worst) gives (N-1)/(N-1) = 1.
      - So: Rank 1 = Best (Highest), Rank N = Worst (Lowest).
      - We want to eliminate Worst (Rank N). So metric should be close to 1.
      - Let's verify: "Rank_pi(Eliminated)". If we elim the lowest vote getter (Rank N), score is 1. Correct.
    """
    n = len(active_names)
    if n <= 1: return { 'fan': 1.0, 'judge': 1.0 } # Default
    
    # 1. Rank Definition: 1 = Best (Highest Value), N = Worst (Lowest Value)
    # scipy rankdata('min') assigns 1 to smallest. 
    # We want 1 to Largest. So rank(-value).
    from scipy.stats import rankdata
    
    # Fan Ranks (1=Highest Share)
    # -shares -> Smallest is Largest Share -> Rank 1
    f_ranks = rankdata([-vote_shares.get(name, 0) for name in active_names], method='average')
    f_rank_map = {name: r for name, r in zip(active_names, f_ranks)}
    
    # Judge Ranks (1=Highest Score)
    j_ranks = rankdata([-judge_scores.get(name, 0) for name in active_names], method='average')
    j_rank_map = {name: r for name, r in zip(active_names, j_ranks)}
    
    # 2. Calculate Metric for Eliminated Set
    # If multiple eliminated, take Average Rank? Or Sum?
    # "E[...]" implies Expected Value over particles. Here we have a set for one particle.
    # Usually we take the average rank of those eliminated in this specific event.
    
    if not eliminated_set:
        return {'fan': 0.0, 'judge': 0.0} # Should not happen
        
    avg_f_rank = np.mean([f_rank_map.get(e, n) for e in eliminated_set])
    avg_j_rank = np.mean([j_rank_map.get(e, n) for e in eliminated_set])
    
    # Formula: (Rank - 1) / (N - 1)
    # But wait, if Rank 1 = Best, Rank N = Worst.
    # We want to eliminate Rank N.
    # So (Rank - 1)/(N-1) -> (N-1)/(N-1) = 1. Perfect.
    # If we eliminate Rank 1 (Best), (1-1)/(N-1) = 0. Perfect.
    
    fan_align = (avg_f_rank - 1) / (n - 1)
    judge_align = (avg_j_rank - 1) / (n - 1)
    
    return {
        'fan_align': max(0.0, min(1.0, fan_align)),
        'judge_align': max(0.0, min(1.0, judge_align))
    }

class FullBenchmark:
    def __init__(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
        self.dp = DataProcessor(csv_path)
        # Optimal parameters from Q4 Optimization
        self.acdw = ACDWRule(p_concave=0.55, lambda_min=0.60, lambda_max=0.80, protection_level=0)
        self.stats = []

    def get_baseline_rules(self, season):
        # Return the actual rule used in that season
        return self.dp.get_rule_type(season)

    def run(self):
        seasons = sorted(self.dp.get_seasons())
        print(f"Benchmarking ACDW-B3 vs Baselines on {len(seasons)} seasons...")
        
        smc = SMCInverse(self.dp)
        
        # We need the CompetitionRules engine for baselines
        from core.competition_rules import CompetitionRules
        base_rules = CompetitionRules()
        
        for season in seasons:
            # print(f"Processing Season {season}...")
            
            def callback(s, w, particles, judge_scores, active_names, event):
                n_particles = len(particles)
                n_elim = len(event.eliminated_set)
                
                # Rule Accumulators
                acc = defaultdict(lambda: {'fan': 0.0, 'judge': 0.0})
                
                for p in particles:
                    # 1. Shares
                    x_values = {name: p.x.get(name, 0) for name in active_names}
                    max_x = max(x_values.values()) if x_values else 0
                    exp_x = {name: np.exp(x - max_x) for name, x in x_values.items()}
                    total = sum(exp_x.values())
                    p_shares = {name: v/total for name, v in exp_x.items()} if total > 0 else {}
                    
                    # --- ACDW-B3 ---
                    outcome_acdw = self.acdw.compute_outcome(judge_scores, p_shares, n_eliminated=n_elim)
                    align_acdw = calculate_alignment(active_names, judge_scores, p_shares, outcome_acdw.eliminated)
                    acc['ACDW-B3']['fan'] += align_acdw['fan_align']
                    acc['ACDW-B3']['judge'] += align_acdw['judge_align']
                    
                    # --- PERCENTAGE ---
                    # Logic: Score = J_share + F_share. Elim = Lowest Score.
                    # Simplified: We use base_rules but need to handle multi-elim manually or rely on its output
                    # Let's do simple manual calc for speed/consistency with ACDW logic
                    # J share
                    total_j = sum(judge_scores.values())
                    j_shares = {k: v/total_j for k, v in judge_scores.items()}
                    scores_pct = {k: j_shares[k] + p_shares[k] for k in active_names}
                    sorted_pct = sorted(scores_pct.items(), key=lambda x: x[1])
                    elim_pct = set(x[0] for x in sorted_pct[:n_elim])
                    align_pct = calculate_alignment(active_names, judge_scores, p_shares, elim_pct)
                    acc['PERCENTAGE']['fan'] += align_pct['fan_align']
                    acc['PERCENTAGE']['judge'] += align_pct['judge_align']
                    
                    # --- RANK ---
                    # Logic: Score = Rank_J + Rank_F. Elim = Highest Sum (Lowest Rank).
                    # Rank 1 = Best.
                    from scipy.stats import rankdata
                    # rankdata gives 1 for smallest. We want 1 for LARGEST value.
                    # so rank(-value)
                    r_j = rankdata([-judge_scores[k] for k in active_names], method='min')
                    r_f = rankdata([-p_shares[k] for k in active_names], method='min')
                    # Map back to names
                    rank_sum = {}
                    for i, name in enumerate(active_names):
                        rank_sum[name] = r_j[i] + r_f[i]
                    # Elim: Largest RankSum (Worst)
                    sorted_rank = sorted(rank_sum.items(), key=lambda x: x[1], reverse=True)
                    elim_rank = set(x[0] for x in sorted_rank[:n_elim])
                    align_rank = calculate_alignment(active_names, judge_scores, p_shares, elim_rank)
                    acc['RANK']['fan'] += align_rank['fan_align']
                    acc['RANK']['judge'] += align_rank['judge_align']
                    
                    # --- RANK + SAVE ---
                    # Same as Rank, but Bottom 2 (or N+1) trigger Save
                    # Candidate Pool: Bottom (n_elim + 1)
                    # Note: Simplified Logic. Real saves are complex.
                    # We assume Judge saves Higher Judge Score
                    n_bottom = n_elim + 1
                    if n_bottom <= len(active_names):
                        candidates = [x[0] for x in sorted_rank[:n_bottom]] # Worst N+1
                        # Save 1: The one with highest Judge Score
                        saved = max(candidates, key=lambda k: judge_scores[k])
                        elim_save = set(c for c in candidates if c != saved)
                    else:
                        elim_save = set(x[0] for x in sorted_rank[:n_elim])
                        
                    align_save = calculate_alignment(active_names, judge_scores, p_shares, elim_save)
                    acc['RANK_SAVE']['fan'] += align_save['fan_align']
                    acc['RANK_SAVE']['judge'] += align_save['judge_align']

                # Average and Store
                for rule, data in acc.items():
                    self.stats.append({
                        'Season': season,
                        'Week': w,
                        'Rule': rule,
                        'Fan_Align': data['fan'] / n_particles,
                        'Judge_Align': data['judge'] / n_particles
                    })
                
            smc.run_season(season, verbose=False, step_callback=callback)
            print(f"Season {season} done.")

    def compute_final_score(self):
        df = pd.DataFrame(self.stats)
        
        # Group by Rule
        summary = df.groupby('Rule')[['Fan_Align', 'Judge_Align']].mean()
        
        # Calculate Score
        w_fan = 0.3
        w_judge = 0.7
        summary['Capped_Fan'] = summary['Fan_Align'].apply(lambda x: min(x, 0.75))
        summary['Score'] = w_fan * summary['Capped_Fan'] + w_judge * summary['Judge_Align']
        
        print("\n=== Comprehensive Rule Benchmark (All 34 Seasons) ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(summary[['Fan_Align', 'Judge_Align', 'Capped_Fan', 'Score']].sort_values('Score', ascending=False))
        
        return summary

if __name__ == "__main__":
    bm = FullBenchmark()
    bm.run()
    bm.compute_final_score()
