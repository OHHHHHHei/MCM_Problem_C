"""
Controversy Analysis V2: Judge-Level Scoring (Plan B)

使用评委级分数统计"垫底次数"，实现更精确的争议分数计算。

公式 (Plan B):
Score = (L / (J × W)) × (1 - P/N)

其中:
- L = 评委级垫底次数 (某评委给出的分数在该周该评委所有选手中最低)
- J = 评委数量
- W = 参赛周数
- P = 最终名次
- N = 赛季参赛人数
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_processor import DataProcessor


# 题目给定的4个争议案例
KNOWN_CASES = {
    "Jerry Rice": {"season": 2, "issue": "5 weeks lowest judge, runner-up"},
    "Billy Ray Cyrus": {"season": 4, "issue": "6 weeks lowest judge, 5th place"},
    "Bristol Palin": {"season": 11, "issue": "12 times lowest judge, 3rd place"},
    "Bobby Bones": {"season": 27, "issue": "Consistently low judge, champion"},
}


class ControversyAnalyzerV2:
    """评委级争议分析器"""
    
    def __init__(self, csv_path: str = '2026_MCM_Problem_C_Data.csv'):
        self.dp = DataProcessor(csv_path)
        self.results = []
        
    def count_judge_lowest(self, season: int, name: str) -> Tuple[int, int, int]:
        """
        统计选手在评委级别的垫底次数
        
        Returns:
            (L, total_opportunities, weeks_participated)
            L = 垫底次数
            total_opportunities = J × W (总评分机会)
            weeks_participated = 参赛周数
        """
        contestants = self.dp.get_contestants_in_season(season)
        
        # 找到目标选手
        target = None
        for c in contestants:
            if c.name == name:
                target = c
                break
        
        if not target:
            return 0, 0, 0
        
        target_weeks = target.weekly_scores.keys()
        
        L = 0  # 垫底次数
        total_opportunities = 0  # 总评分机会
        weeks_participated = 0
        
        for week in sorted(target_weeks):
            # 获取该周活跃选手
            active = self.dp.get_active_contestants(season, week)
            
            if target not in active:
                continue
            
            weeks_participated += 1
            
            # 收集所有选手的评委分数
            # judge_scores[judge_idx] = {contestant_name: score}
            judge_scores = defaultdict(dict)
            
            for c in active:
                scores = c.weekly_scores.get(week, [])
                for j, score in enumerate(scores):
                    if score and score > 0:
                        judge_scores[j][c.name] = score
            
            # 对每个评委检查是否垫底
            for j, scores_dict in judge_scores.items():
                if name not in scores_dict:
                    continue
                
                total_opportunities += 1
                
                # 该评委给出的最低分
                min_score = min(scores_dict.values())
                
                # 如果目标选手是最低分
                if scores_dict[name] == min_score:
                    # 检查是否只有一人最低（不是并列）
                    # 或者我们可以宽松处理：只要是最低就算
                    L += 1
        
        return L, total_opportunities, weeks_participated
    
    def compute_controversy_score_v2(self, L: int, J_W: int, P: int, N: int) -> float:
        """
        计算争议分数 (Plan B)
        
        Score = (L / (J × W)) × (1 - P/N)
        """
        if J_W == 0 or N == 0:
            return 0.0
        
        lowest_rate = L / J_W
        placement_factor = 1 - (P / N)
        
        return lowest_rate * placement_factor
    
    def analyze_all_seasons(self):
        """分析所有赛季的所有选手"""
        print("="*70)
        print("CONTROVERSY ANALYSIS V2: JUDGE-LEVEL SCORING")
        print("="*70)
        
        seasons = self.dp.get_seasons()
        
        for season in seasons:
            contestants = self.dp.get_contestants_in_season(season)
            N = len(contestants)
            
            print(f"\nSeason {season}: {N} contestants")
            
            for c in contestants:
                L, J_W, W = self.count_judge_lowest(season, c.name)
                P = c.placement if c.placement and c.placement < 99 else N
                
                score = self.compute_controversy_score_v2(L, J_W, P, N)
                
                is_known = c.name in KNOWN_CASES and KNOWN_CASES[c.name]["season"] == season
                
                self.results.append({
                    'name': c.name,
                    'season': season,
                    'final_place': P,
                    'n_contestants': N,
                    'L_lowest_count': L,
                    'total_opportunities': J_W,
                    'weeks_participated': W,
                    'lowest_rate': L / J_W if J_W > 0 else 0,
                    'controversy_score_v2': score,
                    'is_known_case': is_known,
                })
                
                if is_known:
                    print(f"  ★ {c.name}: L={L}, J×W={J_W}, P={P}, Score={score:.4f}")
    
    def generate_ranking(self):
        """生成排名报告"""
        df = pd.DataFrame(self.results)
        
        # 按争议分数排序
        df = df.sort_values('controversy_score_v2', ascending=False).reset_index(drop=True)
        
        # 保存完整排名
        os.makedirs('output', exist_ok=True)
        df.to_csv('output/controversy_ranking_v2.csv', index=False)
        print(f"\nSaved full ranking to: output/controversy_ranking_v2.csv")
        
        # 显示Top-15
        print("\n" + "="*70)
        print("TOP-15 MOST CONTROVERSIAL CONTESTANTS (V2 Formula)")
        print("="*70)
        
        for i, row in df.head(15).iterrows():
            marker = "★ KNOWN" if row['is_known_case'] else ""
            print(f"{i+1:2}. {row['name']:25} (S{row['season']:2}) | "
                  f"L={row['L_lowest_count']:2} | P={row['final_place']:2} | "
                  f"Score={row['controversy_score_v2']:.4f} {marker}")
        
        # 检查已知案例排名
        print("\n" + "="*70)
        print("KNOWN CASES RANKING VERIFICATION")
        print("="*70)
        
        for name, info in KNOWN_CASES.items():
            matches = df[(df['name'] == name) & (df['season'] == info['season'])]
            if not matches.empty:
                idx = matches.index[0]
                row = matches.iloc[0]
                in_top10 = "✓ YES" if idx < 10 else "✗ NO"
                print(f"{name}:")
                print(f"  Rank: #{idx+1}")
                print(f"  L (times lowest): {row['L_lowest_count']}")
                print(f"  Issue stated: {info['issue']}")
                print(f"  Score: {row['controversy_score_v2']:.4f}")
                print(f"  In Top-10: {in_top10}")
                print()
        
        return df
    
    def run(self):
        """运行完整分析"""
        self.analyze_all_seasons()
        return self.generate_ranking()


if __name__ == '__main__':
    analyzer = ControversyAnalyzerV2()
    df = analyzer.run()
