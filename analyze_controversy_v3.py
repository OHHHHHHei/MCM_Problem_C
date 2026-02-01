"""
Controversy Analysis V3: Two-Dimensional Framework

实现双维度争议框架：
- S1 (民粹指数 / Populist Score): 评审差但名次好
- S2 (遗珠指数 / Robbed Score): 评审好但被早淘汰

公式:
S1 = max(0, R_pct_avg - P_pct) × (1 - σ_norm)
S2 = max(0, Z_last) × max(0.5, 1 + Z_avg)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_processor import DataProcessor


# 题目给定的4个争议案例
KNOWN_CASES = {
    "Jerry Rice": {"season": 2, "expected_type": "S1", "issue": "5 weeks lowest, runner-up"},
    "Billy Ray Cyrus": {"season": 4, "expected_type": "S1", "issue": "6 weeks lowest, 5th place"},
    "Bristol Palin": {"season": 11, "expected_type": "S1", "issue": "12 times lowest, 3rd place"},
    "Bobby Bones": {"season": 27, "expected_type": "S1", "issue": "Consistently low, champion"},
}


class ControversyAnalyzerV3:
    """双维度争议分析器"""
    
    def __init__(self, csv_path: str = '2026_MCM_Problem_C_Data.csv'):
        self.dp = DataProcessor(csv_path)
        self.results = []
        
    def compute_weekly_ranks(self, season: int) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        计算每位选手每周的评审排名
        
        Returns:
            {contestant_name: [(week, rank, total_score), ...]}
        """
        contestants = self.dp.get_contestants_in_season(season)
        events = self.dp.get_elimination_events(season)
        
        if not events:
            return {}
        
        weeks = sorted(set(e.week for e in events))
        contestant_ranks = defaultdict(list)
        
        for week in weeks:
            # 获取活跃选手
            active = self.dp.get_active_contestants(season, week)
            
            if not active:
                continue
            
            # 计算每位活跃选手的总分
            scores = []
            for c in active:
                total = self.dp.get_weekly_total_score(c, week)
                if total and total > 0:
                    scores.append((c.name, total))
            
            if not scores:
                continue
            
            # 按分数降序排名
            scores.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (name, score) in enumerate(scores, 1):
                contestant_ranks[name].append((week, rank, score))
        
        return contestant_ranks
    
    def compute_z_scores(self, season: int) -> Dict[str, List[Tuple[int, float]]]:
        """
        计算每位选手每周的 Z-Score
        
        Returns:
            {contestant_name: [(week, z_score), ...]}
        """
        contestants = self.dp.get_contestants_in_season(season)
        events = self.dp.get_elimination_events(season)
        
        if not events:
            return {}
        
        weeks = sorted(set(e.week for e in events))
        contestant_zscores = defaultdict(list)
        
        for week in weeks:
            active = self.dp.get_active_contestants(season, week)
            
            if not active:
                continue
            
            # 收集所有分数
            scores = []
            name_score = {}
            for c in active:
                total = self.dp.get_weekly_total_score(c, week)
                if total and total > 0:
                    scores.append(total)
                    name_score[c.name] = total
            
            if len(scores) < 2:
                continue
            
            # 计算 Z-Score
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            if std_score == 0:
                continue
            
            for name, score in name_score.items():
                z = (score - mean_score) / std_score
                contestant_zscores[name].append((week, z))
        
        return contestant_zscores
    
    def compute_s1_populist(self, name: str, season: int, 
                            weekly_ranks: Dict, N: int, P: int) -> Tuple[float, Dict]:
        """
        计算 S1 (民粹指数)
        
        S1 = max(0, R_pct_avg - P_pct) × (1 - σ_norm)
        
        Returns:
            (score, details)
        """
        if name not in weekly_ranks or not weekly_ranks[name]:
            return 0.0, {}
        
        ranks = [r[1] for r in weekly_ranks[name]]  # 提取排名
        W = len(ranks)
        
        if W == 0 or N <= 1:
            return 0.0, {}
        
        # 平均排名百分位 (0=最好, 1=最差)
        R_avg = np.mean(ranks)
        R_pct = (R_avg - 1) / (N - 1)
        
        # 最终名次百分位
        P_pct = (P - 1) / (N - 1)
        
        # 排名标准差归一化
        sigma_rank = np.std(ranks)
        sigma_norm = sigma_rank / (N - 1) if N > 1 else 0
        
        # S1 公式
        gap = R_pct - P_pct
        consistency = 1 - sigma_norm
        
        S1 = max(0, gap) * consistency
        
        details = {
            'R_avg': R_avg,
            'R_pct': R_pct,
            'P_pct': P_pct,
            'gap': gap,
            'sigma_rank': sigma_rank,
            'sigma_norm': sigma_norm,
            'consistency': consistency,
            'weeks': W,
            'ranks': ranks,
        }
        
        return S1, details
    
    def compute_s2_robbed(self, name: str, season: int,
                          weekly_zscores: Dict, P: int, N: int) -> Tuple[float, Dict]:
        """
        计算 S2 (遗珠指数)
        
        S2 = max(0, Z_last) × max(0.5, 1 + Z_avg)
        
        Returns:
            (score, details)
        """
        if name not in weekly_zscores or not weekly_zscores[name]:
            return 0.0, {}
        
        zscores = weekly_zscores[name]
        
        if not zscores:
            return 0.0, {}
        
        # 最后一周的 Z-Score
        last_week, Z_last = max(zscores, key=lambda x: x[0])
        
        # 赛季平均 Z-Score
        Z_avg = np.mean([z[1] for z in zscores])
        
        # S2 公式
        Z_last_term = max(0, Z_last)
        Z_avg_term = max(0.5, 1 + Z_avg)
        
        S2 = Z_last_term * Z_avg_term
        
        # 如果是冠军（没有被淘汰），S2 应该为 0
        if P == 1:
            S2 = 0.0
        
        details = {
            'Z_last': Z_last,
            'last_week': last_week,
            'Z_avg': Z_avg,
            'Z_last_term': Z_last_term,
            'Z_avg_term': Z_avg_term,
            'all_zscores': [(w, round(z, 3)) for w, z in zscores],
        }
        
        return S2, details
    
    def analyze_all_seasons(self):
        """分析所有赛季"""
        print("="*75)
        print("CONTROVERSY ANALYSIS V3: TWO-DIMENSIONAL FRAMEWORK")
        print("="*75)
        print("\nFormulas:")
        print("  S1 (Populist) = max(0, R_pct - P_pct) × (1 - σ_norm)")
        print("  S2 (Robbed)   = max(0, Z_last) × max(0.5, 1 + Z_avg)")
        print()
        
        seasons = self.dp.get_seasons()
        
        for season in seasons:
            contestants = self.dp.get_contestants_in_season(season)
            N = len(contestants)
            
            if N == 0:
                continue
            
            print(f"Season {season}: {N} contestants")
            
            # 预计算
            weekly_ranks = self.compute_weekly_ranks(season)
            weekly_zscores = self.compute_z_scores(season)
            
            for c in contestants:
                P = c.placement if c.placement and c.placement < 99 else N
                
                # 计算 S1
                S1, s1_details = self.compute_s1_populist(
                    c.name, season, weekly_ranks, N, P
                )
                
                # 计算 S2
                S2, s2_details = self.compute_s2_robbed(
                    c.name, season, weekly_zscores, P, N
                )
                
                is_known = c.name in KNOWN_CASES and KNOWN_CASES[c.name]["season"] == season
                
                self.results.append({
                    'name': c.name,
                    'season': season,
                    'final_place': P,
                    'n_contestants': N,
                    'S1_populist': S1,
                    'S2_robbed': S2,
                    'R_avg': s1_details.get('R_avg', 0),
                    'sigma_rank': s1_details.get('sigma_rank', 0),
                    'Z_last': s2_details.get('Z_last', 0),
                    'Z_avg': s2_details.get('Z_avg', 0),
                    'weeks': s1_details.get('weeks', 0),
                    'is_known_case': is_known,
                })
                
                if is_known:
                    print(f"  ★ {c.name}: S1={S1:.3f}, S2={S2:.3f} | "
                          f"R_avg={s1_details.get('R_avg', 0):.1f}, P={P}")
    
    def generate_reports(self):
        """生成报告"""
        df = pd.DataFrame(self.results)
        
        os.makedirs('output', exist_ok=True)
        
        # === S1 排名 (民粹指数) ===
        df_s1 = df.sort_values('S1_populist', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*75)
        print("TOP-15 POPULIST SCORE (S1) - 评审差但名次好")
        print("="*75)
        
        for i, row in df_s1.head(15).iterrows():
            marker = "★ KNOWN" if row['is_known_case'] else ""
            print(f"{i+1:2}. {row['name']:25} (S{row['season']:2}) | "
                  f"P={row['final_place']:2} | R_avg={row['R_avg']:.1f} | "
                  f"S1={row['S1_populist']:.3f} {marker}")
        
        # === S2 排名 (遗珠指数) ===
        df_s2 = df.sort_values('S2_robbed', ascending=False).reset_index(drop=True)
        
        print("\n" + "="*75)
        print("TOP-15 ROBBED SCORE (S2) - 评审好但被早淘汰")
        print("="*75)
        
        for i, row in df_s2.head(15).iterrows():
            marker = "★ KNOWN" if row['is_known_case'] else ""
            print(f"{i+1:2}. {row['name']:25} (S{row['season']:2}) | "
                  f"P={row['final_place']:2} | Z_last={row['Z_last']:.2f} | "
                  f"Z_avg={row['Z_avg']:.2f} | S2={row['S2_robbed']:.3f} {marker}")
        
        # === 已知案例验证 ===
        print("\n" + "="*75)
        print("KNOWN CASES VERIFICATION")
        print("="*75)
        
        for name, info in KNOWN_CASES.items():
            matches = df[(df['name'] == name) & (df['season'] == info['season'])]
            if not matches.empty:
                row = matches.iloc[0]
                
                # S1 排名
                s1_rank = df_s1[df_s1['name'] == name].index[0] + 1
                s2_rank = df_s2[df_s2['name'] == name].index[0] + 1
                
                in_s1_top10 = "✓" if s1_rank <= 10 else "✗"
                in_s2_top10 = "✓" if s2_rank <= 10 else "✗"
                
                print(f"\n{name} (S{info['season']}):")
                print(f"  Issue: {info['issue']}")
                print(f"  Expected Type: {info['expected_type']}")
                print(f"  S1 (Populist): {row['S1_populist']:.3f} | Rank #{s1_rank} | Top-10: {in_s1_top10}")
                print(f"  S2 (Robbed):   {row['S2_robbed']:.3f} | Rank #{s2_rank} | Top-10: {in_s2_top10}")
        
        # === 保存 CSV ===
        df.to_csv('output/controversy_v3_full.csv', index=False)
        print(f"\n\nSaved: output/controversy_v3_full.csv")
        
        # 保存分维度 Top-20
        df_s1.head(20).to_csv('output/controversy_v3_top20_s1.csv', index=False)
        df_s2.head(20).to_csv('output/controversy_v3_top20_s2.csv', index=False)
        print("Saved: output/controversy_v3_top20_s1.csv")
        print("Saved: output/controversy_v3_top20_s2.csv")
        
        return df
    
    def run(self):
        """运行完整分析"""
        self.analyze_all_seasons()
        return self.generate_reports()


if __name__ == '__main__':
    analyzer = ControversyAnalyzerV3()
    df = analyzer.run()
