import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.calibration import calibration_curve

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams
from core.competition_rules import CompetitionRules

# Set Plot Style for Academic Publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def plot_consistency():
    # 1. Config
    csv_path = '2026_MCM_Problem_C_Data.csv'
    
    # Mechanistic Best Parameters (High MAP Match, High Weighted)
    # rho=0.9, gamma=0.0, delta=1.2, alpha=7.0
    best_params = ModelParams(
        n_particles=400, 
        rho=0.9,
        gamma=0.0,
        delta=1.2,
        alpha_discriminant=7.0,
        kappa=0.3, # Assuming from previous best
        beta_judge=2.0
    )
    
    print(f"Running model for Calibration Check with params: {best_params}...")
    
    dp = DataProcessor(csv_path)
    model = SMCInverse(dp, best_params)
    rules_engine = CompetitionRules()
    
    # 2. Run All Seasons (to get enough data points)
    # We can skip season 32 (missing data sometimes)
    seasons = [s for s in dp.get_seasons() if s != 32] 
    
    all_probs = []
    all_outcomes = []
    
    print("Simulating elimination probabilities...")
    
    # Use a subset of seasons to save time if needed, but for a good curve we need data.
    # We will run full model (verbose=False)
    results = model.run_all_seasons(seasons=seasons, verbose=False)
    
    # 3. Post-Process: Monte Carlo Simulation for Probabilities
    # For each week, we have Mean/Std of vote shares. 
    # We reconstruct the probability of being eliminated.
    
    for season, res in results.items():
        estimates = res.get('weekly_estimates', {})
        events = dp.get_elimination_events(season)
        elim_map = {e.week: set(e.eliminated_set) for e in events}
        
        for week, week_est in estimates.items():
            if week not in elim_map:
                continue
                
            actual_eliminated = elim_map[week]
            active_contestants = dp.get_active_contestants(season, week)
            active_names = [c.name for c in active_contestants]
            
            # Prepare Judge Scores
            judge_scores = {c.name: dp.get_weekly_total_score(c, week) for c in active_contestants}
            
            # Prepare Distributions
            means = {n: week_est[n]['mean'] for n in active_names}
            stds = {n: week_est[n]['std'] for n in active_names}
            
            # MC Simulation
            n_sims = 1000
            sim_elim_counts = {n: 0 for n in active_names}
            
            # Vectorized MC could be faster, but loop is safer for logic
            # Let's do a simple loop for logic clarity
            for i in range(n_sims):
                # Sample shares
                sim_shares = {}
                for n in active_names:
                    # Clip at 0
                    val = np.random.normal(means[n], stds[n] + 1e-6)
                    sim_shares[n] = max(0.001, val)
                
                # Normalize
                total_s = sum(sim_shares.values())
                sim_shares = {k: v/total_s for k, v in sim_shares.items()}
                
                # Determine Elimination
                # Use simplified rule: Lowest Survival Score = Eliminated
                # (Ignoring Bottom 2 + Judge Save specifics for simplicity of general probability, 
                # or we can use the exact rule if possible. Strict lowest score is a good proxy for risk)
                
                outcome, _ = rules_engine.compute_survival_scores(season, judge_scores, sim_shares)
                # Sort by score
                outcome.sort(key=lambda x: x.survival_score)
                
                # Who is eliminated?
                # Usually 1 person per week
                # If multiple eliminations in reality, we count the bottom N?
                # Let's assume bottom 1 is high risk. 
                # Better: Check if the person is in the "Drop Zone" (Bottom 2 usually)
                # The prompt asks: "Probability of Elimination". 
                # Let's count how often they are the absolute bottom.
                
                if outcome:
                    elim_name = outcome[0].contestant_name
                    sim_elim_counts[elim_name] += 1
            
            # Record Data Points
            for n in active_names:
                prob = sim_elim_counts[n] / n_sims
                is_eliminated = 1 if n in actual_eliminated else 0
                
                all_probs.append(prob)
                all_outcomes.append(is_eliminated)

    # 4. Plotting
    print(f"Collected {len(all_probs)} data points.")
    
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    
    # Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(all_outcomes, all_probs, n_bins=10)
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    
    # Plot model calibration
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2.5, label='Model Posterior')
    
    # Scatter plot (Jittered) for visual density? No, calibration curve is better.
    # Maybe add a histogram of predicted probabilities at the bottom?
    
    ax.set_ylabel('Fraction of Positives (Actual Elimination Rate)', fontsize=14)
    ax.set_xlabel('Mean Predicted Probability (Model Confidence)', fontsize=14)
    ax.set_title('The Consistency Check:\nDoes "90% Risk" really mean 90% death?', fontsize=16, weight='bold', pad=20)
    
    ax.legend(fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Calculate ECE (Expected Calibration Error) or Brier Score?
    # Simple metric: Brier Score
    brier = np.mean((np.array(all_probs) - np.array(all_outcomes))**2)
    ax.text(0.6, 0.1, f'Brier Score: {brier:.3f}\n(Lower is better)', 
            fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    output_path = 'output/consistency_check.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

if __name__ == '__main__':
    plot_consistency()
