"""
Controversial Cases Analysis for Q2 (Fixed API)

分析题目给定的4个争议案例，并自动发现新争议案例。

输出:
- output/controversial_cases_report.txt: 汇总报告
- output/discovered_controversial.csv: 新发现的争议案例
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.data_processor import DataProcessor


@dataclass
class ControversialCase:
    """争议案例数据结构"""
    name: str
    season: int
    final_place: int
    weeks_lowest_judge: int
    issue: str


# 题目给定的4个争议案例
KNOWN_CASES = [
    ControversialCase("Jerry Rice", 2, 2, 5, "5 weeks lowest judge, runner-up"),
    ControversialCase("Billy Ray Cyrus", 4, 5, 6, "6 weeks lowest judge, 5th place"),
    ControversialCase("Bristol Palin", 11, 3, 12, "12 times lowest judge, 3rd place"),
    ControversialCase("Bobby Bones", 27, 1, -1, "Consistently low judge, champion"),
]


class ControversialCaseAnalyzer:
    """争议案例分析器"""
    
    def __init__(self, csv_path: str = '2026_MCM_Problem_C_Data.csv'):
        self.dp = DataProcessor(csv_path)
        self.results = {}
        self.discovery_results = []
        
    def get_contestant_stats(self, season: int, name: str) -> Dict:
        """获取选手在赛季中的统计数据"""
        contestants = self.dp.get_contestants_in_season(season)
        
        # 找到目标选手对象
        target = None
        for c in contestants:
            if c.name == name:
                target = c
                break
        
        if not target:
            return None
        
        events = self.dp.get_elimination_events(season)
        weeks = sorted(set(e.week for e in events))
        
        stats = {
            'name': name,
            'season': season,
            'weeks_participated': 0,
            'weekly_judge_ranks': [],
            'times_lowest_judge': 0,
            'avg_judge_rank': 0,
        }
        
        for week in weeks:
            # 获取该周活跃选手
            active = self.dp.get_active_contestants(season, week)
            
            if target not in active:
                continue
            
            stats['weeks_participated'] += 1
            
            # 获取所有选手分数
            scores = []
            for c in active:
                score = self.dp.get_weekly_total_score(c, week)
                if score is not None and score > 0:
                    scores.append((c.name, score))
            
            if not scores:
                continue
            
            # 排序 (降序)
            scores.sort(key=lambda x: x[1], reverse=True)
            sorted_names = [s[0] for s in scores]
            
            if name in sorted_names:
                judge_rank = sorted_names.index(name) + 1
                stats['weekly_judge_ranks'].append(judge_rank)
                
                # 是否垫底
                if judge_rank == len(sorted_names):
                    stats['times_lowest_judge'] += 1
        
        if stats['weekly_judge_ranks']:
            stats['avg_judge_rank'] = np.mean(stats['weekly_judge_ranks'])
        
        return stats
    
    def get_final_placement(self, season: int, name: str) -> int:
        """获取选手最终名次"""
        contestants = self.dp.get_contestants_in_season(season)
        
        # 找到目标选手
        for c in contestants:
            if c.name == name:
                return c.placement
        
        return -1
    
    def analyze_known_case(self, case: ControversialCase) -> Dict:
        """分析单个已知争议案例"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {case.name} (Season {case.season})")
        print(f"Issue: {case.issue}")
        print('='*60)
        
        stats = self.get_contestant_stats(case.season, case.name)
        
        if not stats or stats['weeks_participated'] == 0:
            print(f"  No data found for {case.name}")
            return None
        
        final_place = self.get_final_placement(case.season, case.name)
        
        result = {
            'name': case.name,
            'season': case.season,
            'stated_final_place': case.final_place,
            'computed_final_place': final_place,
            'issue': case.issue,
            'weeks_participated': stats['weeks_participated'],
            'times_lowest_judge': stats['times_lowest_judge'],
            'avg_judge_rank': stats['avg_judge_rank'],
            'weekly_judge_ranks': stats['weekly_judge_ranks'],
        }
        
        # 计算争议分数
        if stats['weeks_participated'] > 0 and final_place > 0:
            controversy_score = (stats['avg_judge_rank'] - final_place) / stats['weeks_participated']
        else:
            controversy_score = 0
        result['controversy_score'] = controversy_score
        
        print(f"  Weeks Participated: {stats['weeks_participated']}")
        print(f"  Times Lowest Judge: {stats['times_lowest_judge']}")
        print(f"  Avg Judge Rank: {stats['avg_judge_rank']:.1f}")
        print(f"  Final Place: {final_place}")
        print(f"  Weekly Judge Ranks: {stats['weekly_judge_ranks']}")
        print(f"  Controversy Score: {controversy_score:.3f}")
        
        self.results[case.name] = result
        return result
    
    def discover_controversial_cases(self) -> List[Dict]:
        """自动发现新的争议案例"""
        print("\n" + "="*60)
        print("DISCOVERING NEW CONTROVERSIAL CASES")
        print("="*60)
        
        all_contestants = []
        
        # 遍历所有赛季
        seasons = self.dp.get_seasons()
        
        for season in seasons:
            contestants = self.dp.get_contestants_in_season(season)
            
            for c in contestants:
                name = c.name
                
                stats = self.get_contestant_stats(season, name)
                if not stats or stats['weeks_participated'] < 3:
                    continue
                
                final_place = c.placement
                
                if final_place <= 0:
                    continue
                
                controversy_score = (stats['avg_judge_rank'] - final_place) / stats['weeks_participated']
                
                all_contestants.append({
                    'name': name,
                    'season': season,
                    'final_place': final_place,
                    'avg_judge_rank': stats['avg_judge_rank'],
                    'times_lowest_judge': stats['times_lowest_judge'],
                    'weeks_participated': stats['weeks_participated'],
                    'controversy_score': controversy_score,
                })
        
        if not all_contestants:
            print("No data collected.")
            return []
        
        # 转为DataFrame
        df = pd.DataFrame(all_contestants)
        
        # 排除已知案例
        known_names = {c.name for c in KNOWN_CASES}
        df_new = df[~df['name'].isin(known_names)]
        
        # 过滤: 争议分数 > 0 且有垫底经历
        df_positive = df_new[(df_new['controversy_score'] > 0) & (df_new['times_lowest_judge'] > 0)]
        
        # 按争议分数排序
        df_sorted = df_positive.sort_values('controversy_score', ascending=False)
        
        # 取前10
        top_discoveries = df_sorted.head(10)
        
        print("\nTop 10 Newly Discovered Controversial Cases:")
        print("-" * 60)
        for _, row in top_discoveries.iterrows():
            print(f"  {row['name']} (S{row['season']}): "
                  f"Final={row['final_place']}, AvgJudgeRank={row['avg_judge_rank']:.1f}, "
                  f"LowestJudge={row['times_lowest_judge']}x, "
                  f"Score={row['controversy_score']:.3f}")
        
        self.discovery_results = top_discoveries.to_dict('records')
        
        # 保存所有争议案例
        os.makedirs('output', exist_ok=True)
        df_sorted.to_csv('output/discovered_controversial.csv', index=False)
        print(f"\nSaved {len(df_sorted)} controversial cases to output/discovered_controversial.csv")
        
        return self.discovery_results
    
    def export_results(self):
        """导出分析结果"""
        os.makedirs('output', exist_ok=True)
        
        # 汇总报告
        with open('output/controversial_cases_report.txt', 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("CONTROVERSIAL CASES ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("SECTION 1: KNOWN CASES (from problem statement)\n")
            f.write("-"*70 + "\n\n")
            
            for name, data in self.results.items():
                f.write(f"## {name} (Season {data['season']})\n")
                f.write(f"   Issue: {data['issue']}\n")
                f.write(f"   Final Place: {data['computed_final_place']}\n")
                f.write(f"   Weeks Participated: {data['weeks_participated']}\n")
                f.write(f"   Times Lowest Judge Score: {data['times_lowest_judge']}\n")
                f.write(f"   Average Judge Rank: {data['avg_judge_rank']:.1f}\n")
                f.write(f"   Weekly Judge Ranks: {data['weekly_judge_ranks']}\n")
                f.write(f"   Controversy Score: {data['controversy_score']:.3f}\n")
                f.write("\n")
            
            f.write("\nSECTION 2: NEWLY DISCOVERED CONTROVERSIAL CASES\n")
            f.write("-"*70 + "\n\n")
            
            for i, case in enumerate(self.discovery_results[:10], 1):
                f.write(f"{i}. {case['name']} (Season {case['season']})\n")
                f.write(f"   Final Place: {case['final_place']}\n")
                f.write(f"   Avg Judge Rank: {case['avg_judge_rank']:.1f}\n")
                f.write(f"   Times Lowest Judge: {case['times_lowest_judge']}\n")
                f.write(f"   Controversy Score: {case['controversy_score']:.3f}\n\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("METHODOLOGY\n")
            f.write("-"*70 + "\n")
            f.write("Controversy Score = (Avg Judge Rank - Final Place) / Weeks Participated\n\n")
            f.write("Interpretation:\n")
            f.write("- Positive score: Judges ranked them poorly but audience kept them (CONTROVERSY)\n")
            f.write("- Higher score = More controversial\n")
            f.write("- Example: AvgJudgeRank=8, FinalPlace=2, Weeks=10 -> Score=(8-2)/10=0.6\n")
            f.write("="*70 + "\n")
            
        print("\nSaved: output/controversial_cases_report.txt")
        
        # JSON详情
        with open('output/controversial_cases_details.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        print("Saved: output/controversial_cases_details.json")
    
    def run_full_analysis(self):
        """运行完整分析"""
        print("="*60)
        print("CONTROVERSIAL CASES ANALYSIS")
        print("="*60)
        
        # Phase 1: 分析已知案例
        print("\n[PHASE 1] Analyzing Known Cases...")
        for case in KNOWN_CASES:
            self.analyze_known_case(case)
        
        # Phase 2: 发现新案例
        print("\n[PHASE 2] Discovering New Cases...")
        self.discover_controversial_cases()
        
        # Phase 3: 导出结果
        print("\n[PHASE 3] Exporting Results...")
        self.export_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)


if __name__ == '__main__':
    analyzer = ControversialCaseAnalyzer()
    analyzer.run_full_analysis()
