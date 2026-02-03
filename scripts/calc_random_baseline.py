import os
import sys
import numpy as np
from statistics import mean

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor

def calc_baseline():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    dp = DataProcessor(csv_path)
    
    n_values = []
    
    for season in dp.get_seasons():
        events = dp.get_elimination_events(season)
        for e in events:
            # Active contestants count at the time of elimination
            # Active = Survivors + Eliminated
            n_active = len(e.survivors) + len(e.eliminated_set)
            n_values.append(n_active)
            
    avg_n = mean(n_values)
    baseline_top1 = 1 / avg_n
    baseline_top3 = 3 / avg_n
    
    # Also compute average of (1/N) directly, which is mathematically more precise for "Average Probability"
    avg_prob_top1 = mean([1/n for n in n_values])
    avg_prob_top3 = mean([min(3, n)/n for n in n_values])
    
    print(f"Total Elimination Events: {len(n_values)}")
    print(f"Average N (Active Contestants): {avg_n:.4f}")
    print(f"Simple Baseline (1/AvgN): {baseline_top1:.4%}")
    print(f"Precise Average Probability (Mean(1/N)): {avg_prob_top1:.4%} (This should be ~10%)")
    print(f"Precise Top-3 Probability (Mean(3/N)): {avg_prob_top3:.4%} (This should be ~30%)")

if __name__ == "__main__":
    calc_baseline()
