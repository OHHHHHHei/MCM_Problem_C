
import json
import numpy as np
import sys
import os
from collections import defaultdict
from tqdm import tqdm

# Ensure we can import from core
sys.path.append(os.getcwd())

from core.data_processor import DataProcessor
from core.competition_rules import CompetitionRules
from core.smc_inverse import ParticleState
from core.counterfactual import CounterfactualSimulator

def calculate_matrix():
    print("Loading data...")
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    dp = DataProcessor(csv_path)
    
    with open("output/vote_share_estimates.json", "r") as f:
        estimates_data = json.load(f)
        
    rules = CompetitionRules()
    simulator = CounterfactualSimulator(rules, n_particles=100) # 100 repeats for mean share
    
    # Storage for results
    # scenarios: 'RANK', 'PERCENTAGE', 'RANK_WITH_SAVE', 'PCT_WITH_SAVE'
    results = {
        'RANK': {'fan': [], 'judge': [], 'flip': []},
        'PERCENTAGE': {'fan': [], 'judge': [], 'flip': []},
        'RANK_WITH_SAVE': {'fan': [], 'judge': [], 'flip': []},
        'PCT_WITH_SAVE': {'fan': [], 'judge': [], 'flip': []}
    }
    
    print("Running simulations...")
    
    # Sort seasons to ensure correct state reset order
    sorted_seasons = sorted([int(k) for k in estimates_data.keys()])
    
    for season in tqdm(sorted_seasons):
        season_str = str(season)
        weeks_data = estimates_data[season_str]
        
        # RESET SEASON STATE!
        rules.reset_season()
        
        # Get actual elimination events for the whole season
        events = dp.get_elimination_events(season)
        real_eliminated_by_week = {e.week: set(e.eliminated_set) for e in events}
        
        # Sort weeks to ensure correct accumulation order
        sorted_weeks = sorted([int(k) for k in weeks_data.keys()])
        
        for week in sorted_weeks:
            week_str = str(week)
            est_dict = weeks_data[week_str]
            week = int(week_str)
            
            # Skip if unknown real outcome (e.g. final week or data issue)
            if week not in real_eliminated_by_week:
                continue
                
            real_eliminated = real_eliminated_by_week[week]
            n_eliminated = len(real_eliminated)
            
            active = dp.get_active_contestants(season, week)
            active_names = [c.name for c in active]
            judge_scores = {c.name: dp.get_weekly_total_score(c, week) for c in active}
            
            # Construct Mean Particle
            # "estimates" structure assumption: est_dict = {name: {'mean': float, ...}}
            # Need to normalize just in case
            mean_shares = {}
            for name, stats in est_dict.items():
                if name in active_names:
                    mean_shares[name] = stats['mean']
            
            total_share = sum(mean_shares.values())
            if total_share == 0:
                continue
            mean_shares = {k: v/total_share for k, v in mean_shares.items()}
            
            # Create dummy particle that carries these shares
            # Helper in simulator computes shares from particle.x using softmax.
            # We need to reverse engineer x from shares? 
            # Or assume simulator can take raw shares if we bypass?
            # Simulator._compute_shares_from_particle uses particle.x
            # Let's subclass or mock the simulator method or just make a particle with correct x
            # x = log(share)
            
            x_values = {name: np.log(share + 1e-10) for name, share in mean_shares.items()}
            dummy_particle = ParticleState(
                mu={},
                x=x_values, # This is key for simulator
                weight=1.0,
                accumulated_shares={n: 0.0 for n in active_names}, # Statics snapshot assumption
                accumulated_vote_ranks={n: 0.0 for n in active_names}
            )
            
            # Replicate for probabilistic save
            particles = [dummy_particle] * 100 
            
            # Run 4 Scenarios
            for rule_type in ['RANK', 'PERCENTAGE', 'RANK_WITH_SAVE', 'PCT_WITH_SAVE']:
                
                sim_results = simulator.simulate_single_week(
                    season, week, particles, active_names, judge_scores,
                    rule_type=rule_type, n_eliminated=n_eliminated
                )
                
                # Compute Metrics for this week
                # 1. Flip Rate (Probabilistic)
                # P(Flip) = sum(w * I(sim != real)) / sum(w)
                flip_count = 0
                for sim_set in sim_results:
                    if sim_set != real_eliminated:
                        flip_count += 1
                flip_prob = flip_count / 100.0
                
                # 2. Alignment Scores
                # compute_alignment_scores takes list of sets and weights
                # weights are all 1.0/100
                dummy_weights = [1.0/100] * 100
                align_metrics = simulator.compute_alignment_scores(
                    sim_results, particles, active_names, judge_scores
                )
                
                results[rule_type]['fan'].append(align_metrics['fan_alignment'])
                results[rule_type]['judge'].append(align_metrics['judge_alignment'])
                results[rule_type]['flip'].append(flip_prob)
                
    # Aggregate
    print("\n" + "="*50)
    print("FINAL RESULTS TABLE")
    print("="*50)
    print(f"{'Scenario':<20} | {'Align_Fan':<10} | {'Align_Judge':<10} | {'P(Flip)':<10}")
    print("-" * 60)
    
    for rule_type in ['RANK', 'PERCENTAGE', 'RANK_WITH_SAVE', 'PCT_WITH_SAVE']:
        avg_fan = np.mean(results[rule_type]['fan'])
        avg_judge = np.mean(results[rule_type]['judge'])
        avg_flip = np.mean(results[rule_type]['flip'])
        
        # Store for score calculation
        results[rule_type]['avg_fan'] = avg_fan
        results[rule_type]['avg_judge'] = avg_judge
        results[rule_type]['avg_flip'] = avg_flip
        
        print(f"{rule_type:<20} | {avg_fan:.3f}      | {avg_judge:.3f}        | {avg_flip:.1%}")

    print("\n" + "="*50)
    print("OPTIMALITY SCORE CALCULATION (Correction Capacity Model)")
    print("Formula: Score = w_F * Fan + w_J * Judge + w_C * Correction(Flip_Prob)")
    print("Reason: High Flip_Prob = System's active capacity to correct 'Robbed' outcomes.")
    print("-" * 60)
    
    # Parameters prioritizing Active Correction
    w_f = 0.3
    w_j = 0.3
    w_c = 0.4 
    
    print(f"Weights: Fan={w_f}, Judge={w_j}, Correction={w_c}")
    print(f"{'Scenario':<20} | {'Fan':<6} | {'Judge':<6} | {'Correct':<7} | {'Total Score':<10}")
    print("-" * 60)
    
    for rule_type in ['RANK', 'PERCENTAGE', 'RANK_WITH_SAVE', 'PCT_WITH_SAVE']:
        f = results[rule_type]['avg_fan']
        j = results[rule_type]['avg_judge']
        c = results[rule_type]['avg_flip'] # Correction Capacity
        
        score = w_f * f + w_j * j + w_c * c
        
        print(f"{rule_type:<20} | {f:.3f}  | {j:.3f}  | {c:.3f}    | {score:.4f}")

if __name__ == "__main__":
    calculate_matrix()
