"""
分析模块
用于模型验证、赛制公平性分析和可视化
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import os

from data_processor import DataProcessor
from smc_inverse import SMCInverse, ModelParams


@dataclass
class FairnessMetrics:
    """赛制公平性指标"""
    flip_rate: float          # 翻盘率
    populist_bias: float      # 民粹偏差
    skill_reward_ratio: float # 技术回报比


class ResultAnalyzer:
    """结果分析器"""
    
    def __init__(self, model: SMCInverse, results: Dict[int, Dict]):
        """
        初始化分析器
        
        Args:
            model: SMC模型实例
            results: 各赛季运行结果
        """
        self.model = model
        self.results = results
        self.dp = model.dp
    
    def compute_overall_hit_rate(self, top_n: int = 3) -> Dict:
        """
        计算整体预测命中率
        
        Args:
            top_n: 危险区大小
            
        Returns:
            统计指标
        """
        total_predictions = 0
        total_hits_top1 = 0
        total_hits_top3 = 0
        season_hit_rates_top1 = {}
        season_hit_rates_top3 = {}
        
        for season, result in self.results.items():
            predictions = result.get('predictions', [])
            if not predictions:
                continue
            
            hits_top1 = sum(1 for p in predictions if p.get('hit_top1', False))
            hits_top3 = sum(1 for p in predictions if p.get('hit_top3', False))
            season_hit_rates_top1[season] = hits_top1 / len(predictions) if predictions else 0
            season_hit_rates_top3[season] = hits_top3 / len(predictions) if predictions else 0
            
            total_hits_top1 += hits_top1
            total_hits_top3 += hits_top3
            total_predictions += len(predictions)
        
        return {
            'overall_hit_rate_top1': total_hits_top1 / total_predictions if total_predictions > 0 else 0,
            'overall_hit_rate_top3': total_hits_top3 / total_predictions if total_predictions > 0 else 0,
            'total_predictions': total_predictions,
            'total_hits_top1': total_hits_top1,
            'total_hits_top3': total_hits_top3,
            'season_hit_rates_top1': season_hit_rates_top1,
            'season_hit_rates_top3': season_hit_rates_top3
        }

    def compute_consistency_metrics(self) -> Dict:
        """
        计算新的后验一致性指标
        """
        season_metrics = {}
        all_post_consistencies = []
        all_map_matches = []
        
        for season, result in self.results.items():
            metrics = result.get('consistency_metrics', [])
            if not metrics:
                continue
                
            avg_post = np.mean([m['posterior_consistency'] for m in metrics])
            avg_map = np.mean([1.0 if m['map_consistent'] else 0.0 for m in metrics])
            
            season_metrics[season] = {
                'posterior_consistency': avg_post,
                'map_match_rate': avg_map
            }
            
            all_post_consistencies.extend([m['posterior_consistency'] for m in metrics])
            all_map_matches.extend([1.0 if m['map_consistent'] else 0.0 for m in metrics])
            
        return {
            'overall_posterior_consistency': np.mean(all_post_consistencies) if all_post_consistencies else 0.0,
            'overall_map_match_rate': np.mean(all_map_matches) if all_map_matches else 0.0,
            'season_metrics': season_metrics
        }
    
    def identify_controversies(self, likelihood_threshold: float = -2.0) -> List[Dict]:
        """
        识别争议事件 (低似然淘汰)
        
        低似然意味着模型认为这是一个"意外"淘汰
        
        Args:
            likelihood_threshold: 对数似然阈值
            
        Returns:
            争议事件列表
        """
        controversies = []
        
        for season, result in self.results.items():
            log_likelihoods = result.get('log_likelihoods', [])
            
            for ll_info in log_likelihoods:
                mean_ll = ll_info.get('mean_log_likelihood', 0)
                if mean_ll < likelihood_threshold:
                    controversies.append({
                        'season': season,
                        'week': ll_info['week'],
                        'eliminated': ll_info['eliminated'],
                        'log_likelihood': mean_ll,
                        'surprise_score': -mean_ll  # 越大越意外
                    })
        
        # 按意外程度排序
        controversies.sort(key=lambda x: x['surprise_score'], reverse=True)
        return controversies
    
    def compute_flip_rate(self, 
                          season: int,
                          target_rule: str) -> float:
        """
        计算翻盘率: 切换规则后淘汰结果改变的比例
        
        Args:
            season: 赛季号
            target_rule: 目标规则 ('PERCENTAGE' 或 'RANK')
            
        Returns:
            翻盘率
        """
        result = self.results.get(season, {})
        weekly_estimates = result.get('weekly_estimates', {})
        
        if not weekly_estimates:
            return 0.0
        
        # 获取原始淘汰结果
        events = self.dp.get_elimination_events(season)
        original_eliminated = {e.week: e.eliminated for e in events}
        
        flips = 0
        total = 0
        
        for week, estimates in weekly_estimates.items():
            if week not in original_eliminated:
                continue
            
            original_elim = original_eliminated[week]
            total += 1
            
            # 获取选手分数
            active = self.dp.get_active_contestants(season, week)
            if not active:
                continue
            
            judge_scores = {c.name: self.dp.get_weekly_total_score(c, week) for c in active}
            vote_shares = {name: est['mean'] for name, est in estimates.items()}
            
            # 使用目标规则计算
            from competition_rules import CompetitionRules
            rules = CompetitionRules()
            
            # 临时覆盖规则判断
            original_rule_type = rules._get_rule_type(season)
            
            if target_rule == 'PERCENTAGE':
                # 使用百分比规则
                survival_scores, _ = rules.compute_survival_scores(10, judge_scores, vote_shares)
            else:
                # 使用排名规则
                survival_scores, _ = rules.compute_survival_scores(1, judge_scores, vote_shares)
            
            if survival_scores:
                min_score = min(s.survival_score for s in survival_scores)
                predicted_elim = next(
                    s.contestant_name for s in survival_scores 
                    if s.survival_score == min_score
                )
                
                if predicted_elim != original_elim:
                    flips += 1
        
        return flips / total if total > 0 else 0.0
    
    def compute_populist_bias(self, season: int) -> float:
        """
        计算民粹偏差
        
        式(181): Bias = E[Rank(π_survivors) - Rank(J_survivors)]
        
        正值表示偏向高人气选手，负值表示偏向高技术选手
        """
        result = self.results.get(season, {})
        weekly_estimates = result.get('weekly_estimates', {})
        
        if not weekly_estimates:
            return 0.0
        
        biases = []
        
        for week, estimates in weekly_estimates.items():
            active = self.dp.get_active_contestants(season, week)
            if len(active) < 2:
                continue
            
            # 获取分数和估计份额
            scores = {c.name: self.dp.get_weekly_total_score(c, week) for c in active}
            shares = {name: est['mean'] for name, est in estimates.items() if name in scores}
            
            if len(shares) < 2:
                continue
            
            # 计算排名
            sorted_by_score = sorted(scores.items(), key=lambda x: -x[1])
            sorted_by_share = sorted(shares.items(), key=lambda x: -x[1])
            
            score_ranks = {name: i+1 for i, (name, _) in enumerate(sorted_by_score)}
            share_ranks = {name: i+1 for i, (name, _) in enumerate(sorted_by_share)}
            
            # 计算幸存者的rank差异
            events = self.dp.get_elimination_events(season)
            eliminated_this_week = [e.eliminated for e in events if e.week == week]
            survivors = [name for name in scores if name not in eliminated_this_week]
            
            for survivor in survivors:
                if survivor in score_ranks and survivor in share_ranks:
                    bias = share_ranks[survivor] - score_ranks[survivor]
                    biases.append(bias)
        
        return np.mean(biases) if biases else 0.0
    
    def analyze_rule_fairness(self) -> Dict:
        """
        分析不同规则的公平性
        """
        percentage_seasons = [s for s in self.results.keys() if 3 <= s <= 27]
        rank_seasons = [s for s in self.results.keys() if s <= 2 or s >= 28]
        
        analysis = {
            'percentage_rule': {
                'seasons': percentage_seasons,
                'avg_hit_rate': 0,
                'avg_populist_bias': 0,
            },
            'rank_rule': {
                'seasons': rank_seasons,
                'avg_hit_rate': 0,
                'avg_populist_bias': 0,
            }
        }
        
        # 百分比规则分析
        if percentage_seasons:
            hit_rates = [self.results[s].get('hit_rate_top3', 0) for s in percentage_seasons]
            biases = [self.compute_populist_bias(s) for s in percentage_seasons]
            analysis['percentage_rule']['avg_hit_rate'] = np.mean(hit_rates)
            analysis['percentage_rule']['avg_populist_bias'] = np.mean(biases)
        
        # 排名规则分析
        if rank_seasons:
            hit_rates = [self.results[s].get('hit_rate_top3', 0) for s in rank_seasons]
            biases = [self.compute_populist_bias(s) for s in rank_seasons]
            analysis['rank_rule']['avg_hit_rate'] = np.mean(hit_rates)
            analysis['rank_rule']['avg_populist_bias'] = np.mean(biases)
        
        return analysis
    
    def get_top_vote_getters(self, season: int, week: int, top_n: int = 3) -> List[Tuple[str, float]]:
        """获取某周投票份额最高的选手"""
        result = self.results.get(season, {})
        estimates = result.get('weekly_estimates', {}).get(week, {})
        
        sorted_estimates = sorted(
            [(name, est['mean']) for name, est in estimates.items()],
            key=lambda x: -x[1]
        )
        
        return sorted_estimates[:top_n]
    
    def export_results(self, output_dir: str = '.'):
        """导出分析结果到JSON文件"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 整体统计
        overall = self.compute_overall_hit_rate()
        
        # 争议事件
        controversies = self.identify_controversies()
        
        # 规则公平性
        fairness = self.analyze_rule_fairness()
        
        # 一致性指标
        consistency = self.compute_consistency_metrics()
        
        report = {
            'overall_statistics': overall,
            'consistency_metrics': consistency,
            'top_controversies': controversies[:20],  # 前20个争议事件
            'rule_fairness_analysis': fairness
        }
        
        output_path = os.path.join(output_dir, 'analysis_results.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"Results exported to {output_path}")
        return report


def generate_text_report(analyzer: ResultAnalyzer) -> str:
    """生成文本报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("SMC-Inverse Model Analysis Report")
    lines.append("=" * 60)
    
    # 整体统计
    overall = analyzer.compute_overall_hit_rate()
    lines.append("\n## Overall Prediction Performance")
    lines.append(f"Total Predictions: {overall['total_predictions']}")
    lines.append(f"Total Hits (Top-1): {overall['total_hits_top1']}")
    lines.append(f"Total Hits (Top-3): {overall['total_hits_top3']}")
    lines.append(f"Overall Top-1 Hit Rate: {overall['overall_hit_rate_top1']:.2%}")
    lines.append(f"Overall Top-3 Hit Rate: {overall['overall_hit_rate_top3']:.2%}")

    # 一致性指标
    consistency = analyzer.compute_consistency_metrics()
    lines.append("\n## Consistency & Reconstructability")
    lines.append(f"Overall Posterior Consistency Probability: {consistency['overall_posterior_consistency']:.2%}")
    lines.append(f"Overall MAP Match Rate: {consistency['overall_map_match_rate']:.2%}")
    lines.append("(Note: 'Posterior Consistency' = weighted % of particles agreeing with reality)")
    lines.append("(Note: 'MAP Match Rate' = % of eliminations correctly reconstructed using point estimates)")
    
    # 按赛季分解
    lines.append("\n## Season-by-Season Hit Rates")
    for season in sorted(overall['season_hit_rates_top1'].keys()):
        rule_type = "Percentage" if 3 <= season <= 27 else "Rank"
        rate_top1 = overall['season_hit_rates_top1'][season]
        rate_top3 = overall['season_hit_rates_top3'][season]
        lines.append(f"  Season {season} ({rule_type}): Top-1={rate_top1:.2%}, Top-3={rate_top3:.2%}")
    
    # 规则公平性
    fairness = analyzer.analyze_rule_fairness()
    lines.append("\n## Rule Fairness Comparison")
    lines.append("\nPercentage Rule (S3-S27):")
    lines.append(f"  Average Hit Rate: {fairness['percentage_rule']['avg_hit_rate']:.2%}")
    lines.append(f"  Average Populist Bias: {fairness['percentage_rule']['avg_populist_bias']:.3f}")
    
    lines.append("\nRank Rule (S1-2, S28+):")
    lines.append(f"  Average Hit Rate: {fairness['rank_rule']['avg_hit_rate']:.2%}")
    lines.append(f"  Average Populist Bias: {fairness['rank_rule']['avg_populist_bias']:.3f}")
    
    # 争议事件
    controversies = analyzer.identify_controversies()[:10]
    lines.append("\n## Top 10 Controversial Eliminations")
    for i, c in enumerate(controversies, 1):
        lines.append(f"  {i}. Season {c['season']} Week {c['week']}: {c['eliminated']}")
        lines.append(f"     Surprise Score: {c['surprise_score']:.2f}")
    
    lines.append("\n" + "=" * 60)
    
    return "\n".join(lines)


if __name__ == '__main__':
    from smc_inverse import create_model
    
    print("Creating model...")
    model = create_model(n_particles=300)
    
    print("Running selected seasons...")
    # 运行几个代表性赛季
    results = {}
    for season in [1, 10, 20, 27, 28]:
        try:
            results[season] = model.run_season(season, verbose=True)
        except Exception as e:
            print(f"Error on season {season}: {e}")
    
    # 分析
    analyzer = ResultAnalyzer(model, results)
    report = generate_text_report(analyzer)
    print(report)
    
    # 导出
    analyzer.export_results('output')
