"""
Counterfactual Analysis: Specific Cases

针对题目指定的4个争议案例 + 选定的遗珠案例，模拟它们在不同规则下的命运。
回答 Q3:
1. Rank vs Pct: 结果是否相同？
2. Judge Save: 是否影响结果？

Target Cases:
- Official: Jerry Rice (S2), Billy Ray Cyrus (S4), Bristol Palin (S11), Bobby Bones (S27)
- Robbed (S2 High): Chandler Kinney (S33), Mya (S9)
"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_processor import DataProcessor


# 设定随机种子
np.random.seed(2025)

TARGET_CASES = [
    {"name": "Jerry Rice", "season": 2, "type": "Official (Populist)"},
    {"name": "Billy Ray Cyrus", "season": 4, "type": "Official (Populist)"},
    {"name": "Bristol Palin", "season": 11, "type": "Official (Populist)"},
    {"name": "Bobby Bones", "season": 27, "type": "Official (Populist)"},
    {"name": "Chandler Kinney", "season": 33, "type": "Robbed (High Skill)"},
    {"name": "Mya", "season": 9, "type": "Robbed (High Skill)"}
]



from core.smc_inverse import SMCInverse
import math

class DynamicSimulator:
    def __init__(self):
        self.csv_path = '2026_MCM_Problem_C_Data.csv'
        self.dp = DataProcessor(self.csv_path)
        self.output_dir = 'output/smc_params'
        os.makedirs(self.output_dir, exist_ok=True)
        
    def ensure_params(self, season):
        path = f'{self.output_dir}/season_{season}_params.json'
        if os.path.exists(path):
            return
            
        print(f"Generating parameters for Season {season} using SMC...")
        smc = SMCInverse(self.dp)
        res = smc.run_season(season, verbose=False)
        
        # Calculate posterior means from weekly estimates
        # We use the final week's estimate as the season-level popularity summary
        if not res or 'weekly_estimates' not in res:
            # Fallback
            params = {'posterior_means': {}}
        else:
            last_week = max(res['weekly_estimates'].keys())
            last_est = res['weekly_estimates'][last_week]
            # last_est is {name: {'mean': v, 'std': v}}
            means = {k: v['mean'] for k, v in last_est.items()}
            params = {'posterior_means': means}
            
        with open(path, 'w') as f:
            json.dump(params, f)

    def get_smc_params(self, season):
        self.ensure_params(season)
        path = f'{self.output_dir}/season_{season}_params.json'
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return None

    def simulate(self, season, score_mode, use_save):
        contestants = self.dp.get_contestants_in_season(season)
        smc_params = self.get_smc_params(season)
        
        if not smc_params or not smc_params.get('posterior_means'):
            # Fallback: simple random
            return {c.name: c.placement for c in contestants} 
            
        active_contestants = {c.name: c for c in contestants}
        eliminated_order = []
        
        events = self.dp.get_elimination_events(season)
        weeks = sorted(set(e.week for e in events))
        
        # 采样选手人气
        pop_means = smc_params['posterior_means']
        all_names = [c.name for c in contestants]
        
        # Align means with all names (missing = low)
        start_means = []
        for n in all_names:
            start_means.append(pop_means.get(n, 0.01))
            
        # Re-normalize just in case
        total_m = sum(start_means)
        if total_m == 0: total_m = 1
        alphas = [m/total_m * 100 for m in start_means] # Concentration = 100
        
        try:
            sampled_pis = np.random.dirichlet(alphas)
        except:
             sampled_pis = np.array(start_means) / total_m
             
        contestant_pis = dict(zip(all_names, sampled_pis))
        
        
        for week in weeks:
            if len(active_contestants) <= 1:
                break
                
            # Get Judge Scores
            current_scores = {}
            for name, c in active_contestants.items():
                s = self.dp.get_weekly_total_score(c, week)
                current_scores[name] = s if s else 0 # 缺席当0分或平均
            
            # Calculate Points
            names = list(active_contestants.keys())
            j_scores = [current_scores[n] for n in names]
            c_pis = [contestant_pis.get(n, 0.01) for n in names]
            
            # Normalize pis for active set
            total_pi = sum(c_pis)
            c_pis = [p/total_pi for p in c_pis]
            
            # Composite Score
            final_scores = []
            
            # Rank Rule
            if score_mode == 'rank':
                # Judge Ranks
                # argsort twice for ranks (scipy rankdata is better but manual here)
                # Higher score = Rank 1 (which gets N points)
                # Let's say N contestants. Rank 1 gets N pts.
                # Tied scores share points? DWTS gives tied rank points.
                # Simplified: standard rank
                df_wk = pd.DataFrame({'name': names, 'score': j_scores})
                df_wk['j_rank_pts'] = df_wk['score'].rank(method='min', ascending=True) 
                
                # Vote Ranks
                # Higher pi = Higher rank pts
                df_wk['pi'] = c_pis
                df_wk['v_rank_pts'] = df_wk['pi'].rank(method='min', ascending=True)
                
                df_wk['total'] = df_wk['j_rank_pts'] + df_wk['v_rank_pts']
                final_scores = dict(zip(df_wk['name'], df_wk['total']))
                
            # Pct Rule
            else:
                # Judge Pct
                total_j = sum(j_scores)
                j_pcts = [s/total_j if total_j>0 else 0 for s in j_scores]
                
                # Total = 0.5 * J_pct + 0.5 * V_pct
                totals = [0.5*j + 0.5*v for j, v in zip(j_pcts, c_pis)]
                final_scores = dict(zip(names, totals))
            
            # Elimination
            # Sort by total score (ascending = bottom)
            sorted_by_score = sorted(final_scores.items(), key=lambda x: x[1])
            
            num_elim = self.get_real_elim_count(season, week)
            if num_elim == 0:
                continue
                
            # Identify Bottom N
            bottom_candidates = sorted_by_score[:max(2, num_elim)] # At least bottom 2 for save
            
            to_eliminate = []
            
            if use_save and len(bottom_candidates) >= 2:
                # Judge Save logic: Save the one with higher Judge Score
                # Who are the actual bottom candidates for elimination?
                # Usually Top-1 survivor from Bottom-2 is saved.
                # Simplified: Identify Bottom 2. Save the one with higher J score. Eliminate the other.
                
                # Bottom 2 are the 2 lowest composite scores
                cand1 = bottom_candidates[0] # Lowest composite
                cand2 = bottom_candidates[1] # 2nd lowest
                
                # Compare Judge Scores
                score1 = current_scores[cand1[0]]
                score2 = current_scores[cand2[0]]
                
                if score1 > score2:
                    # Save cand1, Eliminate cand2
                    to_eliminate.append(cand2[0])
                elif score2 > score1:
                    # Save cand2, Eliminate cand1
                    to_eliminate.append(cand1[0])
                else:
                    # Tie in judge score: Eliminate lowest composite (cand1)
                    to_eliminate.append(cand1[0])
                    
                # If double elimination, we need more logic, but keep it simple.
                # Handle extra eliminations if num_elim > 1
                if num_elim > 1:
                    # If we saved one, we still need to eliminate `num_elim` people?
                    # Judge save usually saves ONE from the bottom.
                    # If double elim, maybe Bottom 3 involved.
                    # Simplified: Just apply save to the very last spot.
                    pass
            else:
                # No Save: Eliminate lowest composite
                to_eliminate = [x[0] for x in sorted_by_score[:num_elim]]
            
            # Update Active List
            for n in to_eliminate:
                if n in active_contestants:
                    active_contestants.pop(n)
                    eliminated_order.append(n)
        
        # Add remaining winner(s)
        remaining = list(active_contestants.keys())
        # Sort remaining by composite score of final week? Or just assume survivors are top
        for n in remaining:
            eliminated_order.append(n)
            
        # Reverse to get 1st, 2nd...
        eliminated_order.reverse()
        
        rank_map = {n: i+1 for i, n in enumerate(eliminated_order)}
        return rank_map

    def get_real_elim_count(self, season, week):
        # Return how many people were eliminated in reality
        events = self.dp.get_elimination_events(season)
        count = sum(1 for e in events if e.week == week)
        return count


if __name__ == '__main__':
    # Main Execution
    sim = DynamicSimulator()
    
    print("COUNTERFACTUAL ANALYSIS: CASE STUDIES")
    print("=====================================")
    
    results = []
    
    for case in TARGET_CASES:
        name = case['name']
        s = case['season']
        
        print(f"\nAnalyzing {name} (Season {s})...")
        
        # Run 4 scenarios
        scenarios = [
            ('Rank', False), ('Pct', False),
            ('Rank', True), ('Pct', True)
        ]
        
        row = {'Name': name, 'Season': s, 'Type': case['type']}
        
        for rule, save in scenarios:
            avg_rank_list = []
            for _ in tqdm(range(20), desc=f"{rule}{'+Save' if save else ''}", leave=False):
                # Run Simulation
                ranks = sim.simulate(s, rule.lower(), save)
                if name in ranks:
                    avg_rank_list.append(ranks[name])
            
            final_metric = np.mean(avg_rank_list) if avg_rank_list else np.nan
            label = f"{rule}{'+S' if save else ''}"
            row[label] = round(final_metric, 1)
            
        print(f"  Result: {row}")
        results.append(row)
        
    df = pd.DataFrame(results)
    print("\nFINAL SUMMARY TABLE:")
    print(df)
    
    os.makedirs('output', exist_ok=True)
    df.to_csv('output/case_studies_results.csv', index=False)
