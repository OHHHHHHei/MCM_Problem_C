"""
Q4 ACDW-B3 Benchmark Script (Q2-Aligned Monte Carlo Version)

This script uses the EXACT same methodology as Q2's calculate_q2_matrix.py:
1. CounterfactualSimulator for baseline rules (Monte Carlo with 100 particles)
2. compute_alignment_scores for alignment calculation
3. Ensures numerical consistency with Q2 results
"""

import json
import numpy as np
import sys
import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Any
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.acdw_rule import ACDWRule
from core.competition_rules import CompetitionRules
from core.smc_inverse import ParticleState
from core.counterfactual import CounterfactualSimulator


class Q2AlignedBenchmark:
    """
    Benchmark ACDW-B3 against baselines using Q2's exact Monte Carlo methodology.
    """
    
    def __init__(self, n_particles: int = 100):
        self.csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'data', 'raw', '2026_MCM_Problem_C_Data.csv')
        self.json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                      'output', 'vote_share_estimates.json')
        
        self.dp = DataProcessor(self.csv_path)
        self.rules = CompetitionRules()
        self.simulator = CounterfactualSimulator(self.rules, n_particles=n_particles)
        self.acdw = ACDWRule(p_concave=0.55, lambda_min=0.60, lambda_max=0.80, protection_level=0)
        self.n_particles = n_particles
        self.stats = []

    def run(self):
        print(f"Loading Q2 Estimates from {self.json_path}...")
        with open(self.json_path, 'r') as f:
            estimates_data = json.load(f)
            
        print(f"Benchmarking ACDW-B3 vs Baselines (Monte Carlo with {self.n_particles} particles)...")
        
        sorted_seasons = sorted([int(k) for k in estimates_data.keys()])
        
        for season in tqdm(sorted_seasons, desc="Seasons"):
            season_str = str(season)
            weeks_data = estimates_data[season_str]
            
            # Real Elim Data
            events = self.dp.get_elimination_events(season)
            real_elim_map = {e.week: set(e.eliminated_set) for e in events}
            
            sorted_weeks = sorted([int(k) for k in weeks_data.keys()])
            
            for week in sorted_weeks:
                # 1. Prepare Data
                est_dict = weeks_data[str(week)]
                active = self.dp.get_active_contestants(season, week)
                active_names = [c.name for c in active]
                judge_scores = {c.name: self.dp.get_weekly_total_score(c, week) for c in active}
                
                # Reconstruct Shares
                mean_shares = {}
                for name, stats in est_dict.items():
                    if name in active_names:
                        mean_shares[name] = stats['mean']
                total = sum(mean_shares.values())
                if total == 0: 
                    continue
                p_shares = {k: v/total for k, v in mean_shares.items()}
                
                # Check Elim
                if week not in real_elim_map or not real_elim_map[week]:
                    continue
                n_elim = len(real_elim_map[week])
                
                # Create dummy particles (same as Q2)
                x_values = {name: np.log(share + 1e-10) for name, share in p_shares.items()}
                dummy_particle = ParticleState(
                    mu={},
                    x=x_values,
                    weight=1.0,
                    accumulated_shares={n: 0.0 for n in active_names},
                    accumulated_vote_ranks={n: 0.0 for n in active_names}
                )
                particles = [dummy_particle] * self.n_particles
                
                # --- Baselines using Q2's CounterfactualSimulator ---
                for rule_type in ['RANK', 'PERCENTAGE', 'RANK_WITH_SAVE']:
                    sim_results = self.simulator.simulate_single_week(
                        season, week, particles, active_names, judge_scores,
                        rule_type=rule_type, n_eliminated=n_elim
                    )
                    
                    # Use Q2's alignment calculation method
                    align_metrics = self.simulator.compute_alignment_scores(
                        sim_results, particles, active_names, judge_scores
                    )
                    
                    rule_name = rule_type.replace('_WITH_', '_')
                    self.record(season, week, rule_name, {
                        'fan_align': align_metrics['fan_alignment'],
                        'judge_align': align_metrics['judge_alignment']
                    })
                
                # --- ACDW-B3 (Monte Carlo Simulation) ---
                # Run ACDW multiple times with jittered inputs to match Q2 style
                acdw_fan_ranks = []
                acdw_judge_ranks = []
                
                for _ in range(self.n_particles):
                    # Add small jitter to break ties (matching Q2's approach)
                    jittered_shares = {k: v + np.random.uniform(0, 1e-9) for k, v in p_shares.items()}
                    jittered_judge = {k: v + np.random.uniform(0, 1e-6) for k, v in judge_scores.items()}
                    
                    outcome = self.acdw.compute_outcome(jittered_judge, jittered_shares, n_eliminated=n_elim)
                    
                    # Calculate ranks for alignment (same logic as Q2)
                    n_active = len(active_names)
                    sorted_shares = sorted(p_shares.items(), key=lambda x: -x[1])
                    fan_ranks = {name: i+1 for i, (name, _) in enumerate(sorted_shares)}
                    sorted_judge = sorted(judge_scores.items(), key=lambda x: -x[1])
                    judge_ranks_map = {name: i+1 for i, (name, _) in enumerate(sorted_judge)}
                    
                    for elim_name in outcome.eliminated:
                        r_fan = fan_ranks.get(elim_name, n_active)
                        r_judge = judge_ranks_map.get(elim_name, n_active)
                        acdw_fan_ranks.append((r_fan - 1) / (n_active - 1))
                        acdw_judge_ranks.append((r_judge - 1) / (n_active - 1))
                
                if acdw_fan_ranks:
                    self.record(season, week, 'ACDW-B3', {
                        'fan_align': np.mean(acdw_fan_ranks),
                        'judge_align': np.mean(acdw_judge_ranks)
                    })

    def record(self, season, week, rule, metrics):
        self.stats.append({
            'Season': season,
            'Week': week,
            'Rule': rule,
            'Fan_Align': metrics['fan_align'],
            'Judge_Align': metrics['judge_align']
        })

    def print_results(self):
        df = pd.DataFrame(self.stats)
        
        print(f"\nProcessed {len(df)} simulation points.")
            
        summary = df.groupby('Rule')[['Fan_Align', 'Judge_Align']].mean()
        
        # Weighted Score
        w_fan = 0.3
        w_judge = 0.7
        summary['Capped_Fan'] = summary['Fan_Align'].apply(lambda x: min(x, 0.75))
        summary['Score'] = w_fan * summary['Capped_Fan'] + w_judge * summary['Judge_Align']
        
        print("\n=== ACDW-B3 Benchmark (Q2-Aligned Monte Carlo) ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(summary[['Fan_Align', 'Judge_Align', 'Capped_Fan', 'Score']].sort_values('Score', ascending=False))
        
        return summary


if __name__ == "__main__":
    bm = Q2AlignedBenchmark(n_particles=100)
    bm.run()
    bm.print_results()
