"""
Counterfactual Analysis Engine (Q2)
用于执行反事实模拟，比较不同规则下的淘汰结果
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import copy

from .smc_inverse import ParticleState, ModelParams
from .competition_rules import CompetitionRules, SurvivalScore
from .data_processor import Contestant

class CounterfactualSimulator:
    """反事实模拟器"""
    
    def __init__(self, 
                 rules: CompetitionRules,
                 n_particles: int):
        """
        Args:
            rules: 比赛规则引擎
            n_particles: 粒子数 (K)
        """
        self.rules = rules
        self.n_particles = n_particles
        
    def simulate_single_week(self,
                           season: int,
                           week: int,
                           particles: List[ParticleState],
                           active_names: List[str],
                           judge_scores: Dict[str, float],
                           rule_type: str,
                           n_eliminated: int) -> List[Set[str]]:
        """
        模拟单周淘汰结果
        
        Args:
            season: 赛季
            week: 周次
            particles: 粒子集合 (包含 π)
            active_names: 在场选手名单
            judge_scores: 真实评审分 (J)
            rule_type: 反事实规则 ('RANK', 'PERCENTAGE', 'RANK_WITH_SAVE', 'PCT_WITH_SAVE')
            n_eliminated: 需淘汰的人数 (自适应 M_t)
            
        Returns:
            simulated_eliminations: 每个粒子对应的模拟淘汰者集合 [Set(names), ...]
        """
        # 结果列表
        simulated_results = []
        
        # 临时覆盖规则类型
        original_get_rule = self.rules._get_rule_type
        
        # 辅助函数: 强制指定规则
        def mock_get_rule_type(s):
            if rule_type == 'PCT_WITH_SAVE':
                return 'PERCENTAGE' # Base is Pct
            elif rule_type == 'RANK_WITH_SAVE':
                return 'RANK'       # Base is Rank
            return rule_type

        # 如果是带有Save的规则，S28+ 逻辑是两阶段
        use_save = 'SAVE' in rule_type
        
        try:
            # Monkey path
            self.rules._get_rule_type = mock_get_rule_type
            
            for p in particles:
                # 1. 计算生存分 (Base Metric)
                # 注意: 这里需要正确处理累积逻辑
                # 为简化，反事实分析通常基于 "Snapshot" (切片)，
                # 假设该周之前的历史已经由粒子状态 captured (accumulated_shares/ranks)
                # 我们直接使用粒子中的状态
                
                vote_shares = {}
                
                # 根据 Base Rule 准备投票数据
                base_rule = mock_get_rule_type(season)
                
                curr_shares = self._compute_shares_from_particle(p, active_names)
                
                if base_rule == 'PERCENTAGE':
                    # Pct Rule: Share Accumulation
                    for name in active_names:
                        vote_shares[name] = curr_shares.get(name, 0) + p.accumulated_shares.get(name, 0)
                    # Normalize
                    total = sum(vote_shares.values())
                    if total > 0:
                        vote_shares = {k: v/total for k, v in vote_shares.items()}
                        
                elif base_rule == 'RANK':
                    # Rank Rule: Rank Point Accumulation
                    # 传入 1/Points 作为代理，让 compute_survival_scores 重新 Rank
                    points = {}
                    for name in active_names:
                        points[name] = p.accumulated_vote_ranks.get(name, 0)
                        
                        # 加上本周的 Rank
                        # 需要先对 curr_shares 排序算出本周 Rank
                        # ...但这已经在 smc_inverse 中做过了，而且粒子的 accumulated 已经包含了本周？
                        # 不，smc_inverse loop 中是先 update state, 然后 accumulate, 然后 check event.
                        # 所以 particle passed in SHOULD have accumulated values up to this week.
                        pass
                    
                    # 构造 input
                    # 注意: competition_rules.compute_survival_scores 会对输入再次 Rank
                    # 我们希望 Points 越小越好。
                    # 传入 1.0 / (Points + epsilon)
                    vote_shares = {
                        name: 1.0 / (points.get(name, 0) + 1e-6)
                        for name in active_names
                    }
                
                # 计算生存分
                survival_scores, bottom_two = self.rules.compute_survival_scores(
                    season, judge_scores, vote_shares
                )
                
                eliminated_set = set()
                
                # 2. 确定淘汰者
                if use_save and bottom_two: 
                    # Judge Save 逻辑 (Wrapper)
                    # Bottom 2 已经由 compute_survival_scores (基于 Base Rule) 返回
                    b1, b2 = bottom_two
                    
                    # 决定谁被 Save
                    # 使用 P(Save) = sigmoid(beta * diff)
                    tech_1 = judge_scores.get(b1, 0)
                    tech_2 = judge_scores.get(b2, 0)
                    prob_save_1 = self.rules.compute_judge_save_probability(tech_1, tech_2)
                    
                    # 随机实验
                    if np.random.random() < prob_save_1:
                        # Save b1 -> Eliminate b2
                        elim_cand = b2
                    else:
                        # Save b2 -> Eliminate b1
                        elim_cand = b1
                    
                    # 如果需要淘汰多于1人? 使用 Multi-Elimination Protocol:
                    # 当 M_t >= 2 时:
                    # 1. Auto-Elimination: 直接淘汰 (M_t - 1) 人
                    # 2. Showdown: 剩余人中取 Bottom 2
                    # 3. Judge Save: 救 1 人，淘汰另 1 人
                    # 总淘汰 = (M_t - 1) + 1 = M_t
                    if n_eliminated == 1:
                        eliminated_set.add(elim_cand)
                    else:
                        # Multi-Elimination Protocol
                        sorted_s = sorted(survival_scores, key=lambda x: x.survival_score)
                        
                        # Step 1: Auto-eliminate bottom (M_t - 1)
                        for i in range(n_eliminated - 1):
                            eliminated_set.add(sorted_s[i].contestant_name)
                        
                        # Step 2: Showdown - next 2 lowest form Bottom 2
                        remaining = [s for s in sorted_s if s.contestant_name not in eliminated_set]
                        if len(remaining) >= 2:
                            showdown_1 = remaining[0].contestant_name
                            showdown_2 = remaining[1].contestant_name
                            
                            # Step 3: Judge Save between showdown pair
                            tech_1 = judge_scores.get(showdown_1, 0)
                            tech_2 = judge_scores.get(showdown_2, 0)
                            prob_save_1 = self.rules.compute_judge_save_probability(tech_1, tech_2)
                            
                            if np.random.random() < prob_save_1:
                                eliminated_set.add(showdown_2)  # Save 1, eliminate 2
                            else:
                                eliminated_set.add(showdown_1)  # Save 2, eliminate 1
                        elif len(remaining) == 1:
                            eliminated_set.add(remaining[0].contestant_name)
                            
                else:
                    # Direct Elimination (No Save)
                    sorted_s = sorted(survival_scores, key=lambda x: x.survival_score)
                    for i in range(n_eliminated):
                        eliminated_set.add(sorted_s[i].contestant_name)
                
                simulated_results.append(eliminated_set)
                
        finally:
            # 恢复规则
            self.rules._get_rule_type = original_get_rule
            
        return simulated_results

    def _compute_shares_from_particle(self, particle: ParticleState, active_names: List[str]) -> Dict[str, float]:
        """辅助: 从粒子计算当周份额 (Softmax)"""
        x_values = {name: particle.x.get(name, 0) for name in active_names}
        max_x = max(x_values.values()) if x_values else 0
        exp_x = {name: np.exp(x - max_x) for name, x in x_values.items()}
        total = sum(exp_x.values())
        return {name: v/total for name, v in exp_x.items()} if total > 0 else {}

    def run_full_season_counterfactual(self,
                                     season: int,
                                     target_rule: str) -> Dict:
        """
        运行全赛季反事实推演 (Placeholder)
        实际逻辑需集成在 SMC 循环中
        """
        return {}

    def compute_flip_probability(self, 
                               simulated_results: List[Set[str]],
                               actual_eliminated: Set[str],
                               weights: List[float]) -> float:
        """
        计算翻盘概率
        P(Flip) = Σ w_k * I(Sim_k != Actual)
        """
        flip_prob = 0.0
        total_weight = sum(weights)
        
        for i, sim_set in enumerate(simulated_results):
            # 只要模拟结果集合与真实集合不同，就算翻盘
            if sim_set != actual_eliminated:
                flip_prob += weights[i]
                
        return flip_prob / total_weight if total_weight > 0 else 0.0

    def compute_alignment_scores(self,
                               simulated_results: List[Set[str]],
                               particles: List[ParticleState],
                               active_names: List[str],
                               judge_scores: Dict[str, float]) -> Dict[str, float]:
        """
        计算一致性得分 (Alignment Scores)
        
        Align_Fan = (E[Rank_pi(Elim)] - 1) / (N - 1)
        Align_Judge = (E[Rank_J(Elim)] - 1) / (N - 1)
        """
        n_active = len(active_names)
        if n_active <= 1:
            return {'fan': 1.0, 'judge': 1.0}
            
        total_fan_rank = 0.0
        total_judge_rank = 0.0
        total_weight = 0.0
        
        # 预计算评审分排名 (分数低 -> Rank 大? 
        # 文档定义: Rank(.) 降序排名 (第1名=最高分).
        # 为了 Align Score 越大越好 (Pro-Fan/Pro-Judge):
        # 淘汰 Rank 1 (高分/高票) -> Align = 0 (Bad)
        # 淘汰 Rank N (低分/低票) -> Align = 1 (Good)
        # 所以 Rank 应该是从高到低 (1..N).
        # Align = (Rank - 1) / (N - 1)
        
        # Judge Ranks:
        # 分数越**高**，Rank越小 (1). 分数越**低**，Rank越大 (N).
        sorted_j = sorted(judge_scores.items(), key=lambda x: -x[1]) # Descending
        judge_ranks = {name: i+1 for i, (name, _) in enumerate(sorted_j)}
        
        # 遍历粒子
        for i, elim_set in enumerate(simulated_results):
            p = particles[i]
            w = p.weight
            
            # Fan Ranks (Specific to this particle)
            shares = self._compute_shares_from_particle(p, active_names)
            sorted_shares = sorted(shares.items(), key=lambda x: -x[1]) # Descending
            fan_ranks = {name: k+1 for k, (name, _) in enumerate(sorted_shares)}
            
            # 对集合中每个被淘汰者计算，取平均?
            # 假设 M_t 个淘汰者
            for elim_name in elim_set:
                r_fan = fan_ranks.get(elim_name, n_active)
                r_judge = judge_ranks.get(elim_name, n_active)
                
                # Normalize: (Rank - 1) / (N - 1)
                # Rank 1 -> 0
                # Rank N -> 1
                s_fan = (r_fan - 1) / (n_active - 1)
                s_judge = (r_judge - 1) / (n_active - 1)
                
                total_fan_rank += s_fan * w
                total_judge_rank += s_judge * w
            
            total_weight += w * len(elim_set) # Weight by number of eliminations
            
        return {
            'fan_alignment': total_fan_rank / total_weight if total_weight > 0 else 0.0,
            'judge_alignment': total_judge_rank / total_weight if total_weight > 0 else 0.0
        }

    def check_bobby_bones_hypothesis(self,
                                   season: int,
                                   week: int,
                                   particles: List[ParticleState],
                                   judge_scores: Dict[str, float],
                                   active_names: List[str]) -> Dict[str, float]:
        """
        Bobby Bones 假说验证 (Capping Effect Check)
        
        对比 Bobby Bones 在 Rank Rule vs Percentage Rule 下的被淘汰概率 (PoE)
        
        Returns:
            {
                'PoE_Rank': float,
                'PoE_Pct': float,
                'PoE_Pct_Save': float
            }
        """
        target = "Bobby Bones"
        if target not in active_names:
            return {}
            
        # 1. Rank Rule
        sim_rank = self.simulate_single_week(
            season, week, particles, active_names, judge_scores, 
            rule_type='RANK', n_eliminated=1
        )
        poe_rank = sum(p.weight for i, p in enumerate(particles) if target in sim_rank[i])
        
        # 2. Percentage Rule
        sim_pct = self.simulate_single_week(
            season, week, particles, active_names, judge_scores, 
            rule_type='PERCENTAGE', n_eliminated=1
        )
        poe_pct = sum(p.weight for i, p in enumerate(particles) if target in sim_pct[i])
        
        # 3. Pct + Save
        sim_save = self.simulate_single_week(
            season, week, particles, active_names, judge_scores, 
            rule_type='PCT_WITH_SAVE', n_eliminated=1
        )
        poe_save = sum(p.weight for i, p in enumerate(particles) if target in sim_save[i])
        
        return {
            'PoE_Rank': poe_rank, # Expected High (Capping Effect)
            'PoE_Pct': poe_pct,   # Expected Low (Swamping Effect)
            'PoE_Pct_Save': poe_save # Expected Low (Redundancy)
        }

    def compute_agreement_rate(self,
                              simulated_rank: List[Set[str]],
                              simulated_pct: List[Set[str]],
                              weights: List[float]) -> float:
        """
        计算概率化一致率 (Probabilistic Agreement Rate)
        
        P_t(E_rank = E_pct) = Σ w^{(k)} * I(E_rank^{(k)} = E_pct^{(k)})
        
        Args:
            simulated_rank: Rank Rule 下每个粒子的模拟淘汰集
            simulated_pct: Pct Rule 下每个粒子的模拟淘汰集
            weights: 粒子权重
            
        Returns:
            agreement_rate: [0, 1], 越高表示两规则越一致
        """
        agreement = 0.0
        total_weight = sum(weights)
        
        for i in range(len(weights)):
            if simulated_rank[i] == simulated_pct[i]:
                agreement += weights[i]
                
        return agreement / total_weight if total_weight > 0 else 0.0

    def compute_activation_rate(self,
                               particles: List[ParticleState],
                               active_names: List[str],
                               judge_scores: Dict[str, float],
                               target_name: str,
                               rule_type: str,
                               season: int,
                               week: int) -> float:
        """
        计算 Save 触发率 (Activation Rate)
        
        Act(R) = Pr(Target ∈ Bottom2)
        
        用于检验 "Pct + Save 无效是因为巨星根本进不了 Bottom 2" 的假设
        
        Args:
            target_name: 目标选手 (如 "Bobby Bones")
            rule_type: 基准规则 ('RANK' 或 'PERCENTAGE')
            
        Returns:
            activation_rate: 目标选手进入 Bottom 2 的概率
        """
        if target_name not in active_names:
            return 0.0
            
        activation = 0.0
        total_weight = sum(p.weight for p in particles)
        
        for p in particles:
            # 计算该粒子下的生存分
            vote_shares = self._compute_shares_from_particle(p, active_names)
            
            # 简化: 使用 rules 计算
            survival_scores, bottom_two = self.rules.compute_survival_scores(
                season, judge_scores, vote_shares
            )
            
            # 检查目标是否在 Bottom 2
            if bottom_two and target_name in bottom_two:
                activation += p.weight
                
        return activation / total_weight if total_weight > 0 else 0.0

    def compute_save_marginal_effect(self,
                                    simulated_base: List[Set[str]],
                                    simulated_with_save: List[Set[str]],
                                    actual_eliminated: Set[str],
                                    weights: List[float]) -> float:
        """
        计算 Save 边际效果 (Marginal Effect)
        
        ΔP(Flip) = P(Flip | R + Save) - P(Flip | R)
        
        正值表示 Save 增加了翻盘概率，负值表示减少
        """
        flip_base = self.compute_flip_probability(simulated_base, actual_eliminated, weights)
        flip_save = self.compute_flip_probability(simulated_with_save, actual_eliminated, weights)
        
        return flip_save - flip_base
