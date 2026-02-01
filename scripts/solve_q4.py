"""
Q4 Solution Script: RZF-DS System Simulation
Runs the counterfactual analysis to "Correct" history using the Robust Z-Score Fusion system.
"""

import os
import sys
import argparse
import json
import numpy as np
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams
from core.rzf_policy import RZFPolicy, RZFResult

def load_or_run_smc(dp, seasons, cache_file, particles=300):
    """Load vote shares from cache or run SMC to generate them"""
    if os.path.exists(cache_file):
        print(f"Loading vote shares from {cache_file}...")
        with open(cache_file, 'r') as f:
            raw = json.load(f)
            # Normalize: Extract mean if value is a dict
            normalized = {}
            for s, s_data in raw.items():
                normalized[s] = {}
                for w, w_data in s_data.items():
                    normalized[s][w] = {}
                    for n, v in w_data.items():
                        if isinstance(v, dict) and 'mean' in v:
                            normalized[s][w][n] = float(v['mean'])
                        else:
                            normalized[s][w][n] = float(v)
            return normalized
    
    print("Generating vote shares via SMC (this may take a while)...")
    params = ModelParams(n_particles=particles) 
    
    model = SMCInverse(dp, params)
    results = model.run_all_seasons(seasons, verbose=True)
    
    # Extract mean estimates
    vote_shares = {}
    for s, res in results.items():
        vote_shares[str(s)] = {}
        for w, ests in res['weekly_estimates'].items():
            vote_shares[str(s)][str(w)] = {
                name: float(v['mean']) for name, v in ests.items()
            }
            
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(vote_shares, f, indent=2)
        
    return vote_shares

def run_q4_simulation(args):
    print("="*60)
    print("DWTS Q4: RZF-DS System Counterfactual Simulation")
    print("="*60)
    
    dp = DataProcessor(args.data)
    available_seasons = dp.get_seasons()
    
    target_seasons = args.seasons if args.seasons else available_seasons
    
    # Load Vote Shares (\hat{\pi})
    estimates = load_or_run_smc(dp, target_seasons, args.cache, args.particles)
    
    # Initialize Policy
    policy = RZFPolicy(k_saturation=args.k, w_judge=args.wj, w_fan=args.wf)
    
    # Statistics
    stats = {
        'total_weeks': 0,
        'flips': 0,
        'extreme_mismatches_old': 0,
        'extreme_mismatches_new': 0,
        'alignment_judge_old': [],
        'alignment_judge_new': [],
        'bobby_bones_eliminated': False
    }
    
    print(f"\nRunning simulation on {len(target_seasons)} seasons...")
    
    for season in target_seasons:
        s_str = str(season)
        if s_str not in estimates: continue
        
        events = dp.get_elimination_events(season)
        elim_weeks = {e.week for e in events}
        
        if args.verbose:
            print(f"Season {season} Elim Weeks: {sorted(list(elim_weeks))}")
        
        # Accumulator for No-Elimination weeks
        j_acc = defaultdict(float)
        pi_acc = defaultdict(float)
        weeks_in_block = 0
        
        # Get all weeks for this season
        all_weeks = sorted([int(w) for w in estimates[s_str].keys()])
        
        for week in all_weeks:
            w_str = str(week)
            
            # 1. Get Active Data
            # Key Fix: Filter out historical "Reunion Dancers" (eliminated before this week)
            active_raw = dp.get_active_contestants(season, week)
            active_contestants = []
            for c in active_raw:
                elim_w = dp.get_elimination_week(c)
                if elim_w is not None and elim_w < week:
                    continue # Exclude if historically eliminated
                active_contestants.append(c)
                
            if not active_contestants: continue
            
            current_j = {c.name: dp.get_weekly_total_score(c, week) for c in active_contestants}
            current_pi = estimates[s_str][w_str] # {name: share}
            
            # Filter pi to active only
            current_pi = {k: v for k,v in current_pi.items() if k in current_j}
            
            # 2. Accumulate
            for name, val in current_j.items(): j_acc[name] += val
            for name, val in current_pi.items(): pi_acc[name] += val
            weeks_in_block += 1
            
            # 3. Check if Settlement Week (Elimination or Final Week)
            is_elimination_week = week in elim_weeks
            is_last_week = week == all_weeks[-1]
            
            if is_elimination_week or is_last_week:
                # === SETTLEMENT ===
                
                # Filter Accumulator to ONLY currently active contestants
                # This ensures ghosts don't pollute the rankings
                active_names = set(current_j.keys())
                final_j = {k: v for k, v in j_acc.items() if k in active_names}
                final_pi = {k: v for k, v in pi_acc.items() if k in active_names}
                
                utilities = policy.compute_utilities(season, week, final_j, final_pi)
                
                # B. Determine Elimination Count
                event = next((e for e in events if e.week == week), None)
                num_elim = len(event.eliminated_set) if event else 1
                if is_last_week and not event: num_elim = 0 # Finals
                
                if num_elim > 0:
                    stats['total_weeks'] += 1
                    
                    real_eliminated = set(event.eliminated_set)
                    
                    # Run Trifecta
                    sim_eliminated_list, log = policy.resolve_trifecta_protocol(utilities, num_elim)
                    sim_eliminated = set(sim_eliminated_list)
                    
                    # C. Update Stats
                    
                    # Flip?
                    if sim_eliminated != real_eliminated:
                        stats['flips'] += 1
                        if args.verbose:
                            print(f"\nS{season} W{week} [FLIP]:")
                            print(f"  Real Elim: {real_eliminated}")
                            print(f"  RZF Elim:  {sim_eliminated}")
                            print(f"  Log: {log}")
                            
                    # Bobby Bones Check
                    if season == 27 and 'Bobby Bones' in sim_eliminated:
                        stats['bobby_bones_eliminated'] = True
                        print(f"\n[!!!] BOBBY BONES ELIMINATED in S27 Week {week} under RZF-DS!")
                        
                    # Alignment Metrics (Judge) - BOTTOM 3 DEFINITION
                    sorted_j = sorted(final_j.items(), key=lambda x: x[1])
                    bottom_3_names = {x[0] for x in sorted_j[:3]}
                    
                    matches_old = len(real_eliminated.intersection(bottom_3_names))
                    stats['alignment_judge_old'].append(matches_old / len(real_eliminated))
                    
                    matches_new = len(sim_eliminated.intersection(bottom_3_names))
                    stats['alignment_judge_new'].append(matches_new / len(sim_eliminated))

                # Reset Accumulator
                j_acc = defaultdict(float)
                pi_acc = defaultdict(float)
                weeks_in_block = 0
                
                if is_last_week:
                    print(f"\n[S{season} FINALE] RZF Rankings (Finalists Only):")
                    final_ranking = sorted(utilities, key=lambda x: x.survival_utility, reverse=True)
                    for i, r in enumerate(final_ranking):
                        print(f"  {i+1}. {r.name}: U={r.survival_utility:.4f} (J={r.raw_judge_score}, Pi={r.raw_vote_share:.3f}, Z_J={r.robust_z_judge:.2f}, Z_Pi_Sat={r.robust_z_vote_saturated:.2f})")
                    
                    winner = final_ranking[0].name
                    if season == 27:
                        print(f"  -> RZF Champion: {winner}")
                        if 'Bobby' not in winner:
                             print(f"  -> VICTORY: Bobby Bones lost the Championship! (Rank {len(final_ranking)})")
                        else:
                             print("  -> FAILURE: Bobby Bones still won.")
            else:
                if args.verbose:
                    print(f"S{season} W{week}: No elimination. Accumulating to next week.")
                    
    # Report
    print("\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"Total Elimination Weeks Processed: {stats['total_weeks']}")
    if stats['total_weeks'] > 0:
         print(f"Flip Rate (History Correction): {stats['flips'] / stats['total_weeks']:.2%}")
         
         avg_align_old = np.mean(stats['alignment_judge_old'])
         avg_align_new = np.mean(stats['alignment_judge_new'])
         print(f"Judge Alignment (Meritocracy): {avg_align_old:.3f} (Old) -> {avg_align_new:.3f} (New)")
         print(f"Improvement: {(avg_align_new - avg_align_old)/avg_align_old:.2%}")
         
    if 27 in target_seasons:
        status = "ELIMINATED" if stats['bobby_bones_eliminated'] else "SURVIVED"
        print(f"\nBobby Bones (S27) Status: {status}")
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/raw/2026_MCM_Problem_C_Data.csv')
    parser.add_argument('--seasons', nargs='*', type=int, help='Seasons to run')
    parser.add_argument('--particles', type=int, default=300)
    parser.add_argument('--cache', default='output/vote_share_estimates.json')
    parser.add_argument('--k', type=float, default=0.75, help='Saturation K')
    parser.add_argument('--wj', type=float, default=0.5, help='Judge Weight')
    parser.add_argument('--wf', type=float, default=0.5, help='Fan Weight')
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    run_q4_simulation(parse_args())
