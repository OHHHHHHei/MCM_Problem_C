
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

def calculate_alignment(active_names: List[str],
                        judge_scores: Dict[str, float],
                        vote_shares: Dict[str, float],
                        eliminated_set: Set[str]) -> Dict[str, float]:
    n = len(active_names)
    if n <= 1: return { 'fan': 1.0, 'judge': 1.0 }
    
    from scipy.stats import rankdata
    f_ranks = rankdata([-vote_shares.get(name, 0) for name in active_names], method='average')
    f_rank_map = {name: r for name, r in zip(active_names, f_ranks)}
    j_ranks = rankdata([-judge_scores.get(name, 0) for name in active_names], method='average')
    j_rank_map = {name: r for name, r in zip(active_names, j_ranks)}
    
    if not eliminated_set: return {'fan': 0.0, 'judge': 0.0}
        
    avg_f_rank = np.mean([f_rank_map.get(e, n) for e in eliminated_set])
    avg_j_rank = np.mean([j_rank_map.get(e, n) for e in eliminated_set])
    
    fan_align = (avg_f_rank - 1) / (n - 1)
    judge_align = (avg_j_rank - 1) / (n - 1)
    
    return {
        'fan_align': max(0.0, min(1.0, fan_align)),
        'judge_align': max(0.0, min(1.0, judge_align))
    }

class S27Benchmark:
    def __init__(self):
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
        self.dp = DataProcessor(csv_path)
        self.acdw = ACDWRule(p_concave=0.55, lambda_min=0.60, lambda_max=0.80, protection_level=0)
        self.stats = []

    def run(self):
        print("Benchmarking Rules on Season 27 ONLY...")
        smc = SMCInverse(self.dp)
        
        def callback(s, w, particles, judge_scores, active_names, event):
            n_particles = len(particles)
            n_elim = len(event.eliminated_set)
            acc = defaultdict(lambda: {'fan': 0.0, 'judge': 0.0})
            
            for p in particles:
                x_values = {name: p.x.get(name, 0) for name in active_names}
                max_x = max(x_values.values()) if x_values else 0
                exp_x = {name: np.exp(x - max_x) for name, x in x_values.items()}
                total = sum(exp_x.values())
                p_shares = {name: v/total for name, v in exp_x.items()} if total > 0 else {}
                
                # ACDW
                outcome_acdw = self.acdw.compute_outcome(judge_scores, p_shares, n_eliminated=n_elim)
                align_acdw = calculate_alignment(active_names, judge_scores, p_shares, outcome_acdw.eliminated)
                acc['ACDW-B3']['fan'] += align_acdw['fan_align']
                acc['ACDW-B3']['judge'] += align_acdw['judge_align']
                
                # PERCENTAGE
                total_j = sum(judge_scores.values())
                j_shares = {k: v/total_j for k, v in judge_scores.items()}
                scores_pct = {k: j_shares[k] + p_shares[k] for k in active_names}
                sorted_pct = sorted(scores_pct.items(), key=lambda x: x[1])
                elim_pct = set(x[0] for x in sorted_pct[:n_elim])
                align_pct = calculate_alignment(active_names, judge_scores, p_shares, elim_pct)
                acc['PERCENTAGE']['fan'] += align_pct['fan_align']
                acc['PERCENTAGE']['judge'] += align_pct['judge_align']
                
                # RANK
                from scipy.stats import rankdata
                r_j = rankdata([-judge_scores[k] for k in active_names], method='min')
                r_f = rankdata([-p_shares[k] for k in active_names], method='min')
                rank_sum = {name: r_j[i] + r_f[i] for i, name in enumerate(active_names)}
                sorted_rank = sorted(rank_sum.items(), key=lambda x: x[1], reverse=True)
                elim_rank = set(x[0] for x in sorted_rank[:n_elim])
                align_rank = calculate_alignment(active_names, judge_scores, p_shares, elim_rank)
                acc['RANK']['fan'] += align_rank['fan_align']
                acc['RANK']['judge'] += align_rank['judge_align']
                
                # RANK SAVE
                n_bottom = n_elim + 1
                if n_bottom <= len(active_names):
                    candidates = [x[0] for x in sorted_rank[:n_bottom]]
                    saved = max(candidates, key=lambda k: judge_scores[k])
                    elim_save = set(c for c in candidates if c != saved)
                else:
                    elim_save = set(x[0] for x in sorted_rank[:n_elim])
                align_save = calculate_alignment(active_names, judge_scores, p_shares, elim_save)
                acc['RANK_SAVE']['fan'] += align_save['fan_align']
                acc['RANK_SAVE']['judge'] += align_save['judge_align']

            for rule, data in acc.items():
                self.stats.append({'Season': s, 'Week': w, 'Rule': rule, 'Fan_Align': data['fan']/n_particles, 'Judge_Align': data['judge']/n_particles})
            
        smc.run_season(27, verbose=False, step_callback=callback)

    def print_results(self):
        df = pd.DataFrame(self.stats)
        summary = df.groupby('Rule')[['Fan_Align', 'Judge_Align']].mean()
        w_fan = 0.5; w_judge = 0.5
        summary['Capped_Fan'] = summary['Fan_Align'].apply(lambda x: min(x, 0.75))
        summary['Score'] = w_fan * summary['Capped_Fan'] + w_judge * summary['Judge_Align']
        print("\n=== S27 Only Benchmark ===")
        print(summary[['Fan_Align', 'Judge_Align', 'Score']])

if __name__ == "__main__":
    bm = S27Benchmark()
    bm.run()
    bm.print_results()
