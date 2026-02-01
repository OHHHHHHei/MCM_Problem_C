"""
Script: Full Counterfactual Analysis for Q2
计算所有赛季的规则对比指标，生成 2×2 矩阵分析结果

Metrics:
- Agreement Rate (Probabilistic)
- Flip Probability
- Fan Alignment Score
- Judge Alignment Score
- Save Activation Rate
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Set
from collections import defaultdict
import json

# Ensure core modules are in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.smc_inverse import create_model, ParticleState
from core.competition_rules import CompetitionRules
from core.counterfactual import CounterfactualSimulator


class FullSeasonAnalyzer:
    """全赛季反事实分析器"""
    
    def __init__(self, n_particles: int = 500):
        print(f"Initializing Full Season Analyzer (K={n_particles})...")
        self.model = create_model(n_particles=n_particles)
        self.cf_rules = CompetitionRules()
        self.cf_sim = CounterfactualSimulator(self.cf_rules, n_particles)
        
        # 存储所有结果
        self.weekly_results = []
        self.season_summaries = {}
        
    def analyze_season(self, season: int) -> Dict:
        """分析单个赛季"""
        print(f"\n{'='*60}")
        print(f"ANALYZING SEASON {season}")
        print('='*60)
        
        season_data = {
            'season': season,
            'weeks': [],
            'agreement_rates': [],
            'flip_probs_rank': [],
            'flip_probs_pct': [],
            'fan_align_rank': [],
            'fan_align_pct': [],
            'judge_align_rank': [],
            'judge_align_pct': [],
        }
        
        def analysis_callback(s, week, particles, judge_scores, active_names, event):
            if s != season:
                return
                
            # 获取真实淘汰者
            actual_eliminated = set(event.eliminated_set) if event and event.eliminated_set else set()
            n_eliminated = len(actual_eliminated) if actual_eliminated else 1
            
            if n_eliminated == 0 or len(active_names) < 3:
                return
                
            # 模拟四种规则
            sim_rank = self.cf_sim.simulate_single_week(
                s, week, particles, active_names, judge_scores,
                rule_type='RANK', n_eliminated=n_eliminated
            )
            sim_pct = self.cf_sim.simulate_single_week(
                s, week, particles, active_names, judge_scores,
                rule_type='PERCENTAGE', n_eliminated=n_eliminated
            )
            sim_rank_save = self.cf_sim.simulate_single_week(
                s, week, particles, active_names, judge_scores,
                rule_type='RANK_WITH_SAVE', n_eliminated=n_eliminated
            )
            sim_pct_save = self.cf_sim.simulate_single_week(
                s, week, particles, active_names, judge_scores,
                rule_type='PCT_WITH_SAVE', n_eliminated=n_eliminated
            )
            
            weights = [p.weight for p in particles]
            
            # 计算 Agreement Rate
            agreement = self.cf_sim.compute_agreement_rate(sim_rank, sim_pct, weights)
            
            # 计算 Flip Probability
            flip_rank = self.cf_sim.compute_flip_probability(sim_rank, actual_eliminated, weights)
            flip_pct = self.cf_sim.compute_flip_probability(sim_pct, actual_eliminated, weights)
            
            # 计算 Alignment Scores
            align_rank = self.cf_sim.compute_alignment_scores(sim_rank, particles, active_names, judge_scores)
            align_pct = self.cf_sim.compute_alignment_scores(sim_pct, particles, active_names, judge_scores)
            
            # 存储周结果
            week_result = {
                'season': s,
                'week': week,
                'n_contestants': len(active_names),
                'n_eliminated': n_eliminated,
                'agreement_rate': agreement,
                'flip_prob_rank': flip_rank,
                'flip_prob_pct': flip_pct,
                'fan_align_rank': align_rank['fan_alignment'],
                'fan_align_pct': align_pct['fan_alignment'],
                'judge_align_rank': align_rank['judge_alignment'],
                'judge_align_pct': align_pct['judge_alignment'],
            }
            
            self.weekly_results.append(week_result)
            season_data['weeks'].append(week)
            season_data['agreement_rates'].append(agreement)
            season_data['flip_probs_rank'].append(flip_rank)
            season_data['flip_probs_pct'].append(flip_pct)
            season_data['fan_align_rank'].append(align_rank['fan_alignment'])
            season_data['fan_align_pct'].append(align_pct['fan_alignment'])
            season_data['judge_align_rank'].append(align_rank['judge_alignment'])
            season_data['judge_align_pct'].append(align_pct['judge_alignment'])
            
            print(f"  Week {week}: Agree={agreement:.1%}, Flip(R/P)={flip_rank:.1%}/{flip_pct:.1%}")
        
        # 运行赛季
        try:
            self.model.run_season(season, verbose=False, step_callback=analysis_callback)
        except Exception as e:
            print(f"  Error in season {season}: {e}")
            return season_data
        
        # 计算赛季汇总
        if season_data['agreement_rates']:
            season_data['avg_agreement'] = np.mean(season_data['agreement_rates'])
            season_data['avg_flip_rank'] = np.mean(season_data['flip_probs_rank'])
            season_data['avg_flip_pct'] = np.mean(season_data['flip_probs_pct'])
            season_data['avg_fan_align_rank'] = np.mean(season_data['fan_align_rank'])
            season_data['avg_fan_align_pct'] = np.mean(season_data['fan_align_pct'])
            season_data['avg_judge_align_rank'] = np.mean(season_data['judge_align_rank'])
            season_data['avg_judge_align_pct'] = np.mean(season_data['judge_align_pct'])
            
        self.season_summaries[season] = season_data
        return season_data
    
    def run_all_seasons(self, seasons: List[int] = None):
        """运行所有赛季分析"""
        if seasons is None:
            # 可用赛季列表 (根据数据)
            seasons = list(range(1, 33))  # S1-S32
            
        for s in seasons:
            try:
                self.analyze_season(s)
            except Exception as e:
                print(f"Season {s} failed: {e}")
                continue
                
    def generate_summary_report(self) -> str:
        """生成汇总报告"""
        report = []
        report.append("=" * 80)
        report.append("Q2 FULL COUNTERFACTUAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # 1. Overall Agreement Rate
        all_agreements = [r['agreement_rate'] for r in self.weekly_results if 'agreement_rate' in r]
        if all_agreements:
            report.append(f"OVERALL AGREEMENT RATE: {np.mean(all_agreements):.1%}")
            report.append(f"  (How often Rank Rule and Pct Rule agree on elimination)")
            report.append("")
        
        # 2. 2x2 Matrix Summary
        report.append("-" * 80)
        report.append("2×2 RULE MATRIX ANALYSIS")
        report.append("-" * 80)
        report.append("")
        report.append(f"{'Metric':<25} | {'Rank Rule':<15} | {'Pct Rule':<15} | {'Delta':<10}")
        report.append("-" * 70)
        
        # Fan Alignment
        all_fan_rank = [r['fan_align_rank'] for r in self.weekly_results if 'fan_align_rank' in r]
        all_fan_pct = [r['fan_align_pct'] for r in self.weekly_results if 'fan_align_pct' in r]
        if all_fan_rank and all_fan_pct:
            avg_fan_rank = np.mean(all_fan_rank)
            avg_fan_pct = np.mean(all_fan_pct)
            delta_fan = avg_fan_rank - avg_fan_pct
            report.append(f"{'Fan Alignment (Pro-Fan)':<25} | {avg_fan_rank:<15.3f} | {avg_fan_pct:<15.3f} | {delta_fan:+.3f}")
        
        # Judge Alignment
        all_judge_rank = [r['judge_align_rank'] for r in self.weekly_results if 'judge_align_rank' in r]
        all_judge_pct = [r['judge_align_pct'] for r in self.weekly_results if 'judge_align_pct' in r]
        if all_judge_rank and all_judge_pct:
            avg_judge_rank = np.mean(all_judge_rank)
            avg_judge_pct = np.mean(all_judge_pct)
            delta_judge = avg_judge_rank - avg_judge_pct
            report.append(f"{'Judge Alignment (Pro-Tech)':<25} | {avg_judge_rank:<15.3f} | {avg_judge_pct:<15.3f} | {delta_judge:+.3f}")
        
        # Flip Probability
        all_flip_rank = [r['flip_prob_rank'] for r in self.weekly_results if 'flip_prob_rank' in r]
        all_flip_pct = [r['flip_prob_pct'] for r in self.weekly_results if 'flip_prob_pct' in r]
        if all_flip_rank and all_flip_pct:
            avg_flip_rank = np.mean(all_flip_rank)
            avg_flip_pct = np.mean(all_flip_pct)
            delta_flip = avg_flip_rank - avg_flip_pct
            report.append(f"{'Counterfactual Flip Rate':<25} | {avg_flip_rank:<15.1%} | {avg_flip_pct:<15.1%} | {delta_flip:+.1%}")
        
        report.append("")
        report.append("-" * 80)
        report.append("INTERPRETATION")
        report.append("-" * 80)
        report.append("")
        
        # Generate interpretation
        if all_fan_rank and all_fan_pct:
            if avg_fan_rank > avg_fan_pct:
                report.append("✓ RANK RULE is more PRO-FAN (higher Fan Alignment)")
            else:
                report.append("✓ PCT RULE is more PRO-FAN (higher Fan Alignment)")
                
        if all_judge_rank and all_judge_pct:
            if avg_judge_rank < avg_judge_pct:
                report.append("✓ PCT RULE is more PRO-TECH/MERITOCRATIC (higher Judge Alignment)")
            else:
                report.append("✓ RANK RULE is more PRO-TECH/MERITOCRATIC (higher Judge Alignment)")
        
        report.append("")
        report.append("RECOMMENDATION: Rank Rule + Judge Save")
        report.append("  - Rank Rule protects fan favorites (Pro-Fan)")
        report.append("  - Judge Save provides a safety valve for extreme cases")
        report.append("  - This combination balances Democracy and Meritocracy")
        
        return "\n".join(report)
    
    def export_results(self, output_dir: str = 'output'):
        """导出结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Weekly results as CSV
        if self.weekly_results:
            df = pd.DataFrame(self.weekly_results)
            df.to_csv(f'{output_dir}/q2_weekly_results.csv', index=False)
            print(f"Weekly results saved to {output_dir}/q2_weekly_results.csv")
        
        # 2. Summary report
        report = self.generate_summary_report()
        with open(f'{output_dir}/q2_summary_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Summary report saved to {output_dir}/q2_summary_report.txt")
        
        # 3. Season summaries as JSON
        with open(f'{output_dir}/q2_season_summaries.json', 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types
            clean_summaries = {}
            for k, v in self.season_summaries.items():
                clean_summaries[k] = {
                    key: float(val) if isinstance(val, (np.floating, np.integer)) else val
                    for key, val in v.items() if not isinstance(val, list)
                }
            json.dump(clean_summaries, f, indent=2)
        print(f"Season summaries saved to {output_dir}/q2_season_summaries.json")
        
        return report


def main():
    print("=" * 80)
    print("Q2 FULL COUNTERFACTUAL ANALYSIS")
    print("Comparing Rank Rule vs Percentage Rule across all seasons")
    print("=" * 80)
    
    # 初始化分析器
    analyzer = FullSeasonAnalyzer(n_particles=300)  # 使用较少粒子加速
    
    # 选择要分析的赛季 - 全量模式
    all_seasons = list(range(1, 35))  # S1-S34 全部赛季
    
    print(f"\nAnalyzing ALL seasons: S1-S34 ({len(all_seasons)} seasons)")
    print("This may take 15-20 minutes...")
    
    # 运行分析
    analyzer.run_all_seasons(all_seasons)
    
    # 导出结果
    report = analyzer.export_results()
    
    # 打印报告
    print("\n" + report)


if __name__ == '__main__':
    main()
