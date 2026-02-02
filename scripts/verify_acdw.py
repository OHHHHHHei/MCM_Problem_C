
import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse
from core.acdw_rule import ACDWRule

def compute_shares(particle, active_names):
    x_values = {name: particle.x.get(name, 0) for name in active_names}
    max_x = max(x_values.values()) if x_values else 0
    exp_x = {name: np.exp(x - max_x) for name, x in x_values.items()}
    total = sum(exp_x.values())
    return {name: v/total for name, v in exp_x.items()} if total > 0 else {}

class ACDWVerifier:
    def __init__(self):
        self.csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
        self.dp = DataProcessor(self.csv_path)
        # self.acdw is initialized in run_grid_search or locally

    def run_season_test(self, season, verbose=False):
        if verbose: print(f"\nRunning ACDW Verification on Season {season}...")
        smc = SMCInverse(self.dp)
        
        # Store local results
        bobby_probs = []
        flips = []
        
        def callback(s, w, particles, judge_scores, active_names, event):
            simulated_eliminations = []
            bobby_count = 0
            
            for p in particles:
                shares = compute_shares(p, active_names)
                outcome = self.acdw.compute_outcome(judge_scores, shares)
                
                simulated_eliminations.append(outcome.eliminated)
                if 'Bobby Bones' in outcome.eliminated:
                    bobby_count += 1
            
            # Week Stats
            bobby_prob = bobby_count / len(particles)
            bobby_probs.append(bobby_prob)
            
            # Flip
            actual = event.eliminated_set
            week_flips = sum(1 for sim in simulated_eliminations if sim != actual)
            flips.append(week_flips / len(particles))
            
            if verbose:
                print(f"  Week {w}: Bobby Risk={bobby_prob:.1%}")

        # Run SMC
        smc.run_season(season, verbose=False, step_callback=callback)
        
        avg_bobby_risk = np.mean(bobby_probs) if bobby_probs else 0.0
        avg_flip = np.mean(flips) if flips else 0.0
        return avg_bobby_risk, avg_flip

    def run_grid_search(self):
        p_values = [0.55, 0.60, 0.65, 0.70]
        prot_levels = [0, 1, 2] # 0=None, 1=Top1, 2=Top2
        
        results = []
        
        print(f"\n{'p':<6} {'Prot':<6} {'Bobby Risk (S27)':<18} {'Flip Rate':<10}")
        print("-" * 50)
        
        for p in p_values:
            for prot in prot_levels:
                # Setup
                self.acdw = ACDWRule(p_concave=p, protection_level=prot)
                
                # Run S27 (High Populist Risk Season)
                risk, flip = self.run_season_test(27, verbose=False)
                
                print(f"{p:<6.2f} {prot:<6} {risk:<18.1%} {flip:<10.1%}")
                results.append((p, prot, risk, flip))
                
        # Best Result (Maximize Risk for Bobby while keeping some Flip stability?)
        if results:
            best = max(results, key=lambda x: x[2])
            print("\nACDW-B3 Optimization Result:")
            print(f"Optimal Parameters: p={best[0]}, Protection={best[1]}")
            print(f"Result: Bobby Bones Elimination Probability increases to {best[2]:.1%}")
        else:
            print("No results found.")

if __name__ == "__main__":
    verifier = ACDWVerifier()
    verifier.run_grid_search()
