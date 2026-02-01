"""
Script: Verify Capping Effect Hypothesis (Bobby Bones Case Study)
"""

import sys
import os
import pandas as pd
from typing import List, Dict

# Ensure core modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.smc_inverse import create_model
from core.competition_rules import CompetitionRules
from core.counterfactual import CounterfactualSimulator

def main():
    print("=" * 60)
    print("VERIFYING CAPPING COMPHYPOTHESIS: THE BOBBY BONES PARADOX")
    print("=" * 60)
    
    # 1. Initialize Model
    print("Initializing SMC Model (N=500)...")
    model = create_model(n_particles=500)
    
    # 2. Initialize Counterfactual Simulator
    # We use a separate rules instance for simulation
    cf_rules = CompetitionRules()
    cf_sim = CounterfactualSimulator(cf_rules, model.params.n_particles)
    
    # Data collection
    bobby_stats = []
    
    # 3. Define Callback
    def counterfactual_callback(season, week, particles, judge_scores, active_names, event):
        if season != 27: 
            return
            
        # Run Hypothesis Check
        stats = cf_sim.check_bobby_bones_hypothesis(
            season, week, particles, judge_scores, active_names
        )
        
        if stats:
            stats['week'] = week
            bobby_stats.append(stats)
            
            print(f"  Week {week}: PoE(Rank)={stats['PoE_Rank']:.1%}, "
                  f"PoE(Pct)={stats['PoE_Pct']:.1%}, "
                  f"PoE(Pct+Save)={stats['PoE_Pct_Save']:.1%}")

    # 4. Run Season 27
    print("\nRunning Season 27 Simulation...")
    model.run_season(27, verbose=True, step_callback=counterfactual_callback)
    
    # 5. Output Final Report
    report_path = 'output/bobby_verification.txt'
    os.makedirs('output', exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("FINAL REPORT: BOBBY BONES ELIMINATION PROBABILITIES\n")
        f.write("=" * 60 + "\n")
        f.write(f"{'Week':<6} | {'Rank Rule (Anticipated High)':<30} | {'Pct Rule (Actual Low)':<25} | {'Pct+Save (Redundant/Low)':<25}\n")
        f.write("-" * 90 + "\n")
        
        for stat in bobby_stats:
            f.write(f"{stat['week']:<6} | {stat['PoE_Rank']:<30.1%} | {stat['PoE_Pct']:<25.1%} | {stat['PoE_Pct_Save']:<25.1%}\n")

        f.write("-" * 90 + "\n")
        f.write("\nInterpretation:\n")
        f.write("1. If Rank Rule PoE >> Pct Rule PoE: 'Capping Effect' is CONFIRMED.\n")
        f.write("2. If Pct Rule PoE ~ 0%: 'Swamping Effect' is CONFIRMED.\n")
        f.write("3. If Pct+Save PoE ~ Pct PoE: 'Redundancy' is CONFIRMED.\n")
    
    print(f"Results written to {report_path}")
    
    # Also print to console for immediate check
    with open(report_path, 'r', encoding='utf-8') as f:
        print(f.read())

if __name__ == '__main__':
    main()
