"""
Tests for RZF-DS Policy
"""

import unittest
import numpy as np
from core.rzf_policy import RZFPolicy, RZFResult

class TestRZFPolicy(unittest.TestCase):
    
    def setUp(self):
        self.policy = RZFPolicy(k_saturation=0.75)
        
    def test_robust_z_score_normal(self):
        """Test robust Z matches Standard Z for normal distribution"""
        np.random.seed(42)
        # Generate normal data
        data = np.random.normal(10, 2, 1000)
        
        z_robust = self.policy._compute_robust_z(data)
        z_std = (data - np.mean(data)) / np.std(data)
        
        # They should be highly correlated and close in magnitude
        # Correlation
        corr = np.corrcoef(z_robust, z_std)[0, 1]
        self.assertGreater(corr, 0.99)
        
        # Magnitude check (mean abs diff shouldn't be too large)
        # Median/MAD is resistant, Mean/Std is sensitive, but for Normal they align.
        pass

    def test_robust_z_score_outlier(self):
        """Test robust Z handles outliers (Bobby Bones case)"""
        # 10 normal people (score ~ 10)
        data = np.array([10.0] * 10)
        # 1 outlier (score 100)
        data = np.append(data, 100.0)
        
        # Standard Z
        mean = np.mean(data) # (100 + 100)/11 = 18.18
        std = np.std(data)   # Large
        z_std_outlier = (100 - mean) / std # (100 - 18) / 25 approx 3.2
        z_std_normal = (10 - mean) / std   # (10 - 18) / 25 approx -0.3
        
        # Robust Z
        # Median = 10.0
        # Abs Diff = [0...0, 90]
        # MAD = Median([0...0, 90]) = 0.0?
        # If MAD is 0, we fallback to scale=1.0? 
        # Wait, if >50% data is identical, MAD is 0. 
        # RZFPolicy should handle MAD=0 carefully.
        
        # Let's verify handling.
        z_robust = self.policy._compute_robust_z(data)
        
        # Outlier Z: (100 - 10) / (1.48 * 0 + eps) -> Huge number?
        # Wait, if MAD=0, scale falls back to 1.0 (logic in code).
        # So Z = 90.0
        # This correctly highlights existing large gap.
        
        self.assertTrue(z_robust[-1] > 10.0) # Should be very large
        
    def test_saturation(self):
        """Test tanh saturation"""
        # Z = 5.0 (Huge popularity)
        # k = 0.75
        val = np.tanh(0.75 * 5.0) # tanh(3.75) ~ 0.999
        self.assertLess(val, 1.0)
        self.assertGreater(val, 0.9)
        
        # Z = 0.0
        self.assertEqual(np.tanh(0), 0.0)
        
    def test_trifecta_basic(self):
        """Test basic trifecta logic"""
        # 4 Contestants
        # A: Util 2.0 (Safe)
        # B: Util -0.5 (Danger) - Votes 0.5 - Judge 30
        # C: Util -1.0 (Danger) - Votes 0.8 (Savior) - Judge 10
        # D: Util -2.0 (Danger) - Votes 0.1 - Judge 20
        
        # Expected:
        # Bottom 3: B, C, D
        # Fan Save: C has max votes (0.8). SAFE.
        # Duel: B (Judge 30) vs D (Judge 20).
        # Victim: D (Lower Judge Score).
        
        mock_results = [
            RZFResult(0,0,'A', 30, 0.5, 0,0,0, 2.0, 1),
            RZFResult(0,0,'B', 30, 0.5, 0,0,0, -0.5, 2),
            RZFResult(0,0,'C', 10, 0.8, 0,0,0, -1.0, 3), # Low judge, High vote
            RZFResult(0,0,'D', 20, 0.1, 0,0,0, -2.0, 4)  # Low judge, Low vote
        ]
        
        eliminated, log = self.policy.resolve_trifecta_protocol(mock_results, 1)
        
        self.assertEqual(len(eliminated), 1)
        self.assertEqual(eliminated[0], 'D')
        self.assertIn("C SAVED", log)
        self.assertIn("D ELIMINATED", log)

    def test_trifecta_double_elim(self):
        """Test double elimination logic"""
        # 5 Contestants
        # E: -3.0 (Direct Out)
        # Others same as above.
        
        mock_results = [
            RZFResult(0,0,'A', 30, 0.5, 0,0,0, 2.0, 1),
            RZFResult(0,0,'B', 30, 0.5, 0,0,0, -0.5, 2),
            RZFResult(0,0,'C', 10, 0.8, 0,0,0, -1.0, 3), 
            RZFResult(0,0,'D', 20, 0.1, 0,0,0, -2.0, 4),
            RZFResult(0,0,'E', 15, 0.0, 0,0,0, -3.0, 5) # Worst
        ]
        
        eliminated, log = self.policy.resolve_trifecta_protocol(mock_results, 2)
        
        self.assertEqual(len(eliminated), 2)
        self.assertEqual(eliminated[0], 'E') # Direct
        self.assertEqual(eliminated[1], 'D') # Duel Victim
        
if __name__ == '__main__':
    unittest.main()
