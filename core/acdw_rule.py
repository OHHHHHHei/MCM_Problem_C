
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from scipy.stats import spearmanr
from dataclasses import dataclass

@dataclass
class ACDWResult:
    survival_scores: Dict[str, float]
    eliminated: Set[str]
    lambda_weight: float
    rho_consensus: float
    bottom_three: List[str]
    is_protected: bool

class ACDWRule:
    """
    ACDW-B3: Adaptive Concave Diminishing-returns + Weighted + Bottom-3 Save
    """
    
    def __init__(self, 
                 p_concave: float = 0.65,
                 lambda_min: float = 0.60,
                 lambda_max: float = 0.80,
                 protection_level: int = 2):
        self.p = p_concave
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.protection_level = protection_level
        
    def compute_outcome(self,
                       judge_scores: Dict[str, float],
                       vote_shares: Dict[str, float],
                       n_eliminated: int = 1) -> ACDWResult:
        """
        计算单周 ACDW 结果
        """
        active_names = list(judge_scores.keys())
        n = len(active_names)
        if n < 3:
            # Fallback for minimal contestants (Finals)
            return self._compute_simple_outcome(judge_scores, vote_shares, n_eliminated)
            
        # 1. Judge Share (j_i)
        total_j = sum(judge_scores.values())
        j_shares = {k: v/total_j for k, v in judge_scores.items()}
        
        # 2. Fan Share Transformation (Concave g(v) = v^p)
        # Note: Input vote_shares are already normalized shares (pi)
        # We apply transform and re-normalize
        v_transformed = {k: v**self.p for k, v in vote_shares.items()}
        total_v = sum(v_transformed.values())
        f_shares = {k: v/total_v for k, v in v_transformed.items()}
        
        # 3. Dynamic Weighting (Spearman Consensus)
        # Rank: 1 is best (highest score/share)
        # spearmanr ranks: 1 is smallest value? No, spearmanr handles values.
        # We pass raw values. High score/share = High Rank value in scipy?
        # scipy.stats.rankdata assigns 1 to smallest.
        # So "Correlation of Ranks" is same as "Correlation of Values" (monotonic).
        # We want to measure agreement.
        
        j_vec = [j_shares[name] for name in active_names]
        f_vec = [f_shares[name] for name in active_names]
        
        # Correlation
        # Handle constant input (undef correlation)
        if len(set(j_vec)) < 2 or len(set(f_vec)) < 2:
            rho = 0.0
        else:
            rho, _ = spearmanr(j_vec, f_vec)
            if np.isnan(rho): rho = 0.0
            
        # Map rho to lambda
        # rho = 1 (Agree) -> lambda_min (Fan friendly)
        # rho = -1 (Disagree) -> lambda_max (Judge friendly)
        # Formula: lambda = min + (max - min) * (1 - rho)/2
        lambda_t = self.lambda_min + (self.lambda_max - self.lambda_min) * (1 - rho) / 2.0
        
        # 4. Composite Score
        scores = {}
        for name in active_names:
            scores[name] = lambda_t * j_shares[name] + (1 - lambda_t) * f_shares[name]
            
        # 5. Bottom 3
        # Score is Share-like (Higher is better)
        sorted_by_score = sorted(scores.items(), key=lambda x: x[1]) # Low to High
        if len(sorted_by_score) >= 3:
            bottom_3 = [x[0] for x in sorted_by_score[:3]]
        else:
            bottom_3 = [x[0] for x in sorted_by_score] # Should not happen if n>=3
            
        # 6. Judge Save Logic with Constraints
        # Candidate pool: Bottom 3
        # Constraint: "Fan Protection"
        # Use protection_level to determine how many top fans are protected
        
        # Identify Top N Fans
        sorted_by_fan = sorted(f_shares.items(), key=lambda x: -x[1]) # High to Low
        n_protected = min(self.protection_level, len(sorted_by_fan))
        protected_set = set(x[0] for x in sorted_by_fan[:n_protected])
        
        # Filter Bottom 3
        eligible_for_elimination = []
        protected_count = 0
        
        for name in bottom_3:
            if name in protected_set:
                # PROTECTED!
                protected_count += 1
            else:
                eligible_for_elimination.append(name)
                
        eliminated_set = set()
        
        # If everyone is protected? (Possible if n=3, Bottom 3 = All, Top 2 fans in Bottom 3)
        # Fallback: If no one is eligible (rare deadlock), revert to lowest TOTAL score
        if not eligible_for_elimination:
            # Deadlock breaker: eliminate lowest total score among them
            # (Ignore protection to ensure someone goes home)
            target = sorted_by_score[0][0]
            eliminated_set.add(target)
        else:
            # Normal Judge Save: Eliminate Lowest Tech Score among eligible
            # "Eliminate = argmin J_{i,t}"
            sorted_eligible = sorted(eligible_for_elimination, key=lambda n: judge_scores.get(n, 0))
            
            # Eliminate needed amount (usually 1)
            for i in range(min(len(sorted_eligible), n_eliminated)):
                eliminated_set.add(sorted_eligible[i])
                
        return ACDWResult(
            survival_scores=scores,
            eliminated=eliminated_set,
            lambda_weight=lambda_t,
            rho_consensus=rho,
            bottom_three=bottom_3,
            is_protected=(protected_count > 0)
        )

    def _compute_simple_outcome(self, j, v, n_elim):
        # ... logic for finals ...
        # simplified Pct rule
        pass
        # Just return dummy for now as focus is regular season S27
        return ACDWResult({}, set(), 0.5, 0.0, [], False)
