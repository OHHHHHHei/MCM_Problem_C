import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams

def run_sensitivity_analysis():
    print("="*60)
    print("Q1 Sensitivity Analysis: Rho (Memory) vs Gamma (Guide)")
    print("="*60)
    
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    dp = DataProcessor(csv_path)
    
    # Parameter Grid
    # Rho: Memory Decay (0.5=Low Memory, 0.95=High Memory)
    rhos = [0.5, 0.7, 0.8, 0.9, 0.95]
    # Gamma: Judge Guide (0=Independent, 1=High Guide)
    gammas = [0.0, 0.2, 0.5, 0.8, 1.0]
    
    # Test Seasons (Subset for speed)
    # S19 (High controversy switch), S27 (Bobby Bones), S30 (Modern)
    test_seasons = [19, 27, 30] 
    
    results = []
    
    total_runs = len(rhos) * len(gammas)
    run_count = 0
    
    for rho, gamma in product(rhos, gammas):
        run_count += 1
        print(f"Run {run_count}/{total_runs}: rho={rho}, gamma={gamma}...", end="", flush=True)
        
        # Configure Model
        # Using fewer particles for speed if just checking sensitivity
        params = ModelParams(
            rho=rho,
            gamma=gamma,
            n_particles=100  # Smaller for sensitivity sweep
        )
        
        smc = SMCInverse(dp, params)
        
        metrics = {'hit_top3': [], 'map_match': [], 'log_likely': []}
        
        for season in test_seasons:
            run_res = smc.run_season(season, verbose=False)
            if not run_res: continue
            
            # Extract Metrics
            # Hit Rate
            hits = [p['hit_top3'] for p in run_res['predictions']]
            if hits:
                metrics['hit_top3'].append(np.mean(hits))
                
            # Log Likelihood
            lls = [l['mean_log_likelihood'] for l in run_res['log_likelihoods']]
            if lls:
                metrics['log_likely'].append(np.mean(lls))
                
            # MAP Consistency (if available in results)
            # Need to extract from consistency_metrics
            if 'consistency_metrics' in run_res:
                 map_cons = [m['map_consistent'] for m in run_res['consistency_metrics']]
                 if map_cons:
                     metrics['map_match'].append(np.mean(map_cons))
        
        # Average over seasons
        avg_hit = np.mean(metrics['hit_top3']) if metrics['hit_top3'] else 0
        avg_map = np.mean(metrics['map_match']) if metrics['map_match'] else 0
        avg_ll = np.mean(metrics['log_likely']) if metrics['log_likely'] else -999
        
        results.append({
            'rho': rho,
            'gamma': gamma,
            'hit_rate': avg_hit,
            'map_match': avg_map,
            'log_likely': avg_ll
        })
        
        print(f" Hit={avg_hit:.2%}, MAP={avg_map:.2%}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df)
    
    # Save CSV
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'q1_sensitivity.csv'), index=False)
    
    # Plot Heatmaps
    plot_sensitivity(df, output_dir)

def plot_sensitivity(df, output_dir):
    # Pivot for Heatmap
    pivot_hit = df.pivot(index='rho', columns='gamma', values='hit_rate')
    pivot_map = df.pivot(index='rho', columns='gamma', values='map_match')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Hit Rate
    sns.heatmap(pivot_hit, annot=True, fmt=".1%", cmap="viridis", ax=axes[0])
    axes[0].set_title("Sensitivity: Top-3 Hit Rate (Prediction Power)")
    axes[0].set_xlabel("Gamma (Judge Guide)")
    axes[0].set_ylabel("Rho (Memory Decay)")
    
    # Plot 2: MAP Match
    sns.heatmap(pivot_map, annot=True, fmt=".1%", cmap="magma", ax=axes[1])
    axes[1].set_title("Sensitivity: MAP Match Rate (Logical Consistency)")
    axes[1].set_xlabel("Gamma (Judge Guide)")
    axes[1].set_ylabel("Rho (Memory Decay)")
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images', 'q1_sensitivity_rho_gamma.png')
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Heatmap saved to {plot_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
