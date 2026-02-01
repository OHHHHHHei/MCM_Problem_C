"""
RZF-DS Policy Module
Robust Z-Score Fusion with Dynamic Saturation & Trifecta Elimination Protocol
Question 4 Solution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RZFResult:
    """RZF Calculation Result for a single contestant"""
    season: int
    week: int
    name: str
    raw_judge_score: float
    raw_vote_share: float
    robust_z_judge: float
    robust_z_vote_raw: float
    robust_z_vote_saturated: float  # After tanh
    survival_utility: float  # Final fused score
    rank_utility: int  # 1 = Best

class RZFPolicy:
    """
    RZF-DS System Implementation
    
    1. Robust Standardization: (X - Median) / (1.4826 * MAD)
    2. Dynamic Saturation: tanh(k * Z_vote)
    3. Trifecta Protocol: Bottom 3 -> Fan Save -> Duel
    """
    
    def __init__(self, 
                 k_saturation: float = 0.75,
                 w_judge: float = 0.5,
                 w_fan: float = 0.5):
        """
        Args:
            k_saturation: Saturation coefficient for vote shares (default 0.75)
            w_judge: Weight for Judge Utility (default 0.5)
            w_fan: Weight for Fan Utility (default 0.5)
        """
        self.k = k_saturation
        self.w_j = w_judge
        self.w_f = w_fan
        self.consistency_factor = 1.4826  # 1 / Phi^-1(0.75)
        
    def _compute_robust_z(self, values: np.ndarray) -> np.ndarray:
        """
        Compute Robust Z-Score using Median and MAD
        Formula: Z = (X - Median) / (1.4826 * MAD + epsilon)
        """
        if len(values) < 2:
            return np.zeros_like(values)
            
        median = np.median(values)
        abs_diffs = np.abs(values - median)
        mad = np.median(abs_diffs)
        
        # Avoid division by zero
        scale = self.consistency_factor * mad
        if scale < 1e-6:
            scale = 1.0  # Fallback corresponding to std=1.0 if strictly constant
            
        return (values - median) / scale

    def compute_utilities(self, 
                         season: int,
                         week: int,
                         judge_scores: Dict[str, float], 
                         vote_shares: Dict[str, float]) -> List[RZFResult]:
        """
        Compute Survival Utilities for all contestants in a week.
        """
        names = list(judge_scores.keys())
        if not names:
            return []
            
        # Align data arrays
        j_vals = np.array([judge_scores[n] for n in names])
        v_vals = np.array([vote_shares.get(n, 0.0) for n in names])
        
        # 1. Robust Z-Scores
        z_j = self._compute_robust_z(j_vals)
        z_v_raw = self._compute_robust_z(v_vals)
        
        # 2. Dynamic Saturation for Votes (Asymmetric)
        # Goal: Cap "Infinite Popularity" (Z > 0) but keep "Unpopularity" (Z <= 0) linear.
        # This prevents punishing low-popularity contestants with saturation.
        # For Z > 0: tanh(k * Z)
        # For Z <= 0: k * Z (Linearly scaled to match the slope at 0)
        z_v_sat = np.where(
            z_v_raw > 0,
            np.tanh(self.k * z_v_raw),
            self.k * z_v_raw
        )
        
        # 3. Fusion
        # Note: We assume Z_J is NOT saturated (Merit should be linear reward)
        # But Z_V IS saturated (Popularity capped)
        utilities = self.w_j * z_j + self.w_f * z_v_sat
        
        # Create results
        results = []
        # Rank: Higher utility is better. Sort descending.
        sorted_indices = np.argsort(-utilities)
        ranks = np.zeros(len(names), dtype=int)
        for i, idx in enumerate(sorted_indices):
            ranks[idx] = i + 1
            
        for i, name in enumerate(names):
            results.append(RZFResult(
                season=season,
                week=week,
                name=name,
                raw_judge_score=j_vals[i],
                raw_vote_share=v_vals[i],
                robust_z_judge=z_j[i],
                robust_z_vote_raw=z_v_raw[i],
                robust_z_vote_saturated=z_v_sat[i],
                survival_utility=utilities[i],
                rank_utility=ranks[i]
            ))
            
        return results

    def resolve_trifecta_protocol(self, 
                                 utilities: List[RZFResult],
                                 num_to_eliminate: int = 1) -> Tuple[List[str], str]:
        """
        Execute Trifecta Elimination Protocol.
        
        Protocol:
        1. Danger Zone: Bottom 3 (or Bottom N+2 if N>1)
        2. Fan Safety Net: Highest Vote Share in Danger Zone is SAFE.
        3. Redemption Duel: Remaining contestants judged by RAW JUDGE SCORE.
           - Lowest Judge Score goes home.
           
        Args:
            utilities: List of RZFResults
            num_to_eliminate: How many need to go home (default 1)
            
        Returns:
            (eliminated_names, reason_log)
        """
        # Sort by utility ascending (worst first)
        sorted_utils = sorted(utilities, key=lambda x: x.survival_utility)
        n_candidates = len(sorted_utils)
        
        log = []
        eliminated = []
        
        if n_candidates <= num_to_eliminate:
            # Everyone goes home? Rare edge case at finals.
            return [x.name for x in sorted_utils[:num_to_eliminate]], "Automatic Elimination (Small Pool)"
            
        # 1. Define Danger Zone
        # Standard: Bottom 3 for Single Elim.
        # For Double Elim (N=2), logic implies we need to eliminate 2.
        # Trifecta usually implies: 1 Safe, 1 Duel Winner, Others Out?
        # Let's adapt:
        # If N=1: Bottom 3 -> 1 Safe -> 2 Duel -> 1 Out.
        # If N=2: Bottom 4 -> 1 Safe -> 3 Duel? Or Bottom 4 -> ...
        # Based on docs:
        # "Double Elimination: Lowest S directly OUT. Resulting Bottom 3 -> Trifecta for 2nd slot."
        
        # Let's implement the documented logic strictly.
        
        current_pool = sorted_utils[:] # Copy
        
        # Step A: Direct Elimination (if N > 1)
        while num_to_eliminate > 1:
            direct_victim = current_pool.pop(0) # Lowest Score
            eliminated.append(direct_victim.name)
            log.append(f"Double Elim: {direct_victim.name} eliminated directly (Lowest Utility {direct_victim.survival_utility:.3f})")
            num_to_eliminate -= 1
            
        # Now num_to_eliminate == 1. Run Standard Trifecta.
        # Danger Zone = Bottom 3 of remaining
        danger_zone_size = min(3, len(current_pool))
        danger_zone = current_pool[:danger_zone_size]
        safe_zone = current_pool[danger_zone_size:]
        
        log.append(f"Danger Zone (Bottom {danger_zone_size}): {[x.name for x in danger_zone]}")
        
        if danger_zone_size == 1:
            # Only 1 person in danger? Just eliminate.
            victim = danger_zone[0]
            eliminated.append(victim.name)
            log.append(f"Automatic Elim: {victim.name} (Only 1 in Danger Zone)")
            return eliminated, "\n".join(log)

        # Step B: Fan Safety Net
        # Who has highest raw_vote_share in Danger Zone?
        fan_savior = max(danger_zone, key=lambda x: x.raw_vote_share)
        log.append(f"Fan Safety Net: {fan_savior.name} SAVED (Vote Share {fan_savior.raw_vote_share:.4f})")
        
        # Remove Savior from Danger Zone
        duelists = [x for x in danger_zone if x.name != fan_savior.name]
        
        if not duelists:
            # Should not happen if size >= 2
            return eliminated, "\n".join(log)
            
        if len(duelists) == 1:
            # Only 1 left? They go home.
            victim = duelists[0]
            eliminated.append(victim.name)
            log.append(f"Default Elim: {victim.name} (No duel opponent)")
            return eliminated, "\n".join(log)
            
        # Step C: Redemption Duel
        # Tie-breaker: Raw Judge Score
        # If Judge Scores tied? Random or Lower Utility goes.
        # We sort duelists by Judge Score Ascending.
        # Lowest Judge Score is eliminated.
        
        # Sort by Judge Score Ascending
        # Tie-break: Survival Utility Ascending (Secondary)
        duelists_sorted = sorted(duelists, key=lambda x: (x.raw_judge_score, x.survival_utility))
        
        victim = duelists_sorted[0]
        winner = duelists_sorted[1:] # Logically exist
        
        eliminated.append(victim.name)
        log.append(f"Redemption Duel: {victim.name} ELIMINATED vs {[w.name for w in winner]}. (Judge Score {victim.raw_judge_score} vs {[w.raw_judge_score for w in winner]})")
        
        return eliminated, "\n".join(log)

