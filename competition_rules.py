"""
比赛规则模块
实现DWTS不同赛季的评分和淘汰规则
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SurvivalScore:
    """生存分数"""
    contestant_name: str
    judge_score_share: float  # 评审分份额
    fan_vote_share: float  # 粉丝投票份额
    effective_cumulative_share: float  # 有效累积份额
    survival_score: float  # 最终生存分


class CompetitionRules:
    """比赛规则实现"""
    
    def __init__(self, 
                 xi_perturbation: float = 1e-4,  # Rank制平局微扰
                 lambda_decay: float = 0.9,  # 滚存衰减因子
                 judge_save_beta: float = 2.0):  # 评委拯救系数
        """
        初始化规则参数
        
        Args:
            xi_perturbation: Rank制平局时的微扰系数
            lambda_decay: 无淘汰周分数滚存的衰减因子
            judge_save_beta: Judge Save时评委倾向系数
        """
        self.xi = xi_perturbation
        self.lambda_decay = lambda_decay
        self.beta_judge = judge_save_beta
        
        # 存储上周累积份额 (用于滚存)
        self.prev_cumulative_shares: Dict[str, float] = {}
        self.prev_week_had_elimination: bool = True
    
    def reset_season(self):
        """重置赛季状态"""
        self.prev_cumulative_shares = {}
        self.prev_week_had_elimination = True
    
    def update_elimination_status(self, had_elimination: bool):
        """更新淘汰状态"""
        self.prev_week_had_elimination = had_elimination
    
    def compute_effective_cumulative_share(self, 
                                          contestant_name: str,
                                          current_share: float) -> float:
        """
        计算有效累积份额 (处理滚存逻辑)
        
        式(83): 
        - 若上周有淘汰: π̂_t = π_t
        - 若上周无淘汰: π̂_t = (π̂_{t-1} + λ·π_t) / (1 + λ)
        """
        if self.prev_week_had_elimination:
            # 正常周，直接使用当周份额
            effective = current_share
        else:
            # 滚存周，加权平均
            prev = self.prev_cumulative_shares.get(contestant_name, current_share)
            effective = (prev + self.lambda_decay * current_share) / (1 + self.lambda_decay)
        
        # 更新存储
        self.prev_cumulative_shares[contestant_name] = effective
        return effective
    
    def compute_percentage_rule_score(self,
                                      contestant_name: str,
                                      judge_score: float,
                                      total_judge_scores: float,
                                      fan_vote_share: float) -> SurvivalScore:
        """
        计算 Percentage Rule 下的生存分 (S3-S27)
        
        式(94): S^{Pct}_{i,t} = J_{i,t}/Σ_k J_{k,t} + π̂_{i,t}
        """
        judge_share = judge_score / total_judge_scores if total_judge_scores > 0 else 0.0
        effective_share = self.compute_effective_cumulative_share(contestant_name, fan_vote_share)
        survival = judge_share + effective_share
        
        return SurvivalScore(
            contestant_name=contestant_name,
            judge_score_share=judge_share,
            fan_vote_share=fan_vote_share,
            effective_cumulative_share=effective_share,
            survival_score=survival
        )
    
    def compute_rank_rule_score(self,
                                contestant_name: str,
                                judge_rank: int,
                                vote_rank: int,
                                fan_vote_share: float,
                                n_contestants: int) -> SurvivalScore:
        """
        计算 Rank Rule 下的生存分 (S1-2, S28+)
        
        式(100): S^{Rank}_{i,t} = -[Rank(J) + Rank(π̂)] + ξ·π̂_{i,t}
        
        rank越小越好(1最好)，所以取负使得S越大越安全
        """
        effective_share = self.compute_effective_cumulative_share(contestant_name, fan_vote_share)
        
        # 基础排名分 (取负，使得更小的rank得到更高的S)
        base_score = -(judge_rank + vote_rank)
        
        # 加上微扰项处理平局
        survival = base_score + self.xi * effective_share
        
        # 归一化judge_share用于返回
        judge_share = 1.0 - (judge_rank - 1) / (n_contestants - 1) if n_contestants > 1 else 1.0
        
        return SurvivalScore(
            contestant_name=contestant_name,
            judge_score_share=judge_share,
            fan_vote_share=fan_vote_share,
            effective_cumulative_share=effective_share,
            survival_score=survival
        )
    
    def compute_judge_save_probability(self,
                                       contestant_a_cum_score: float,
                                       contestant_b_cum_score: float) -> float:
        """
        计算 Judge Save 时评委拯救A的概率
        
        式(142): P(Save A) = σ(β_judge · (J_cum_A - J_cum_B))
        """
        diff = contestant_a_cum_score - contestant_b_cum_score
        return 1.0 / (1.0 + np.exp(-self.beta_judge * diff))
    
    def identify_bottom_two(self,
                            survival_scores: List[SurvivalScore]) -> Tuple[str, str]:
        """
        识别 Bottom 2 (生存分最低的两人)
        用于 S28+ 的 Judge Save 机制
        """
        sorted_scores = sorted(survival_scores, key=lambda x: x.survival_score)
        if len(sorted_scores) >= 2:
            return (sorted_scores[0].contestant_name, sorted_scores[1].contestant_name)
        elif len(sorted_scores) == 1:
            return (sorted_scores[0].contestant_name, '')
        else:
            return ('', '')
    
    def compute_survival_scores(self,
                                season: int,
                                judge_scores: Dict[str, float],
                                fan_vote_shares: Dict[str, float],
                                cumulative_judge_scores: Optional[Dict[str, float]] = None
                               ) -> Tuple[List[SurvivalScore], Optional[Tuple[str, str]]]:
        """
        计算所有选手的生存分
        
        Args:
            season: 赛季号
            judge_scores: 各选手当周评审分 {name: score}
            fan_vote_shares: 各选手投票份额 {name: share}
            cumulative_judge_scores: 累计评审分 (用于Judge Save)
            
        Returns:
            survival_scores: 各选手生存分
            bottom_two: Bottom 2 (仅S28+返回)
        """
        results = []
        bottom_two = None
        
        rule_type = self._get_rule_type(season)
        
        if rule_type == 'PERCENTAGE':
            # Percentage Rule (S3-S27)
            total_judge = sum(judge_scores.values())
            for name, score in judge_scores.items():
                vote_share = fan_vote_shares.get(name, 0.0)
                results.append(self.compute_percentage_rule_score(
                    name, score, total_judge, vote_share
                ))
        
        elif rule_type in ['RANK', 'RANK_WITH_SAVE']:
            # Rank Rule (S1-2, S28+)
            n = len(judge_scores)
            
            # 计算评审分排名 (加极小噪声打破平局)
            # 噪声量级 1e-6 远小于最小分差0.5，不影响非平局顺序
            # 但能确保平局时不同粒子得到不同排名，符合概率性思想
            jittered_judge = {
                name: score + np.random.uniform(0, 1e-6) 
                for name, score in judge_scores.items()
            }
            sorted_by_judge = sorted(jittered_judge.items(), key=lambda x: -x[1])
            judge_ranks = {name: rank+1 for rank, (name, _) in enumerate(sorted_by_judge)}
            
            # 计算投票份额排名 (同样加噪声)
            jittered_vote = {
                name: share + np.random.uniform(0, 1e-9)
                for name, share in fan_vote_shares.items()
            }
            sorted_by_vote = sorted(jittered_vote.items(), key=lambda x: -x[1])
            vote_ranks = {name: rank+1 for rank, (name, _) in enumerate(sorted_by_vote)}
            
            for name, score in judge_scores.items():
                vote_share = fan_vote_shares.get(name, 0.0)
                results.append(self.compute_rank_rule_score(
                    name,
                    judge_ranks.get(name, n),
                    vote_ranks.get(name, n),
                    vote_share,
                    n
                ))
            
            # S28+ 需要识别 Bottom 2
            if rule_type == 'RANK_WITH_SAVE':
                bottom_two = self.identify_bottom_two(results)
        
        return results, bottom_two
    
    def _get_rule_type(self, season: int) -> str:
        """内部方法：获取规则类型"""
        if season <= 2:
            return 'RANK'
        elif season <= 27:
            return 'PERCENTAGE'
        else:
            return 'RANK_WITH_SAVE'


class LikelihoodCalculator:
    """似然函数计算器"""
    
    def __init__(self, alpha_discriminant: float = 5.0):
        """
        初始化
        
        Args:
            alpha_discriminant: 判别力参数，控制似然函数的陡峭程度
        """
        self.alpha = alpha_discriminant
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid函数"""
        # 防止溢出
        if x > 500:
            return 1.0
        elif x < -500:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x))
    
    def compute_base_elimination_likelihood(self,
                                            eliminated_score: float,
                                            survivor_scores: List[float]) -> float:
        """
        计算基础淘汰似然 (单人淘汰)
        
        式(121): L = Π_{j∈S} σ(α·(S_j - S_e))
        
        要求被淘汰者的分数低于所有幸存者
        """
        likelihood = 1.0
        for s_j in survivor_scores:
            prob = self.sigmoid(self.alpha * (s_j - eliminated_score))
            likelihood *= prob
        return likelihood
    
    def compute_group_elimination_likelihood(self,
                                            eliminated_scores: List[float],
                                            survivor_scores: List[float]) -> float:
        """
        计算集合淘汰似然 (支持多人淘汰)
        
        修正双淘汰周逻辑bug: 不再要求淘汰者之间有大小关系
        
        数学表达:
        L = Π_{e∈E} Π_{s∈S} σ(α·(S_s - S_e))
        
        含义: 每个被淘汰者的分数都低于每个幸存者
        淘汰者内部的相对顺序不影响似然
        """
        likelihood = 1.0
        
        for e_score in eliminated_scores:
            for s_score in survivor_scores:
                # P(幸存者分数 > 被淘汰者分数)
                prob = self.sigmoid(self.alpha * (s_score - e_score))
                likelihood *= prob
        
        return likelihood
    
    def compute_judge_save_likelihood(self,
                                      eliminated_name: str,
                                      saved_name: str,
                                      survival_scores: Dict[str, float],
                                      cumulative_tech_scores: Dict[str, float],
                                      beta_judge: float = 2.0) -> float:
        """
        计算 Judge Save 似然 (两阶段)
        
        式(136):
        L_Save = P(Bottom 2 = {e, s}) × P(Save s | Tech-Score)
        
        Args:
            eliminated_name: 被淘汰者
            saved_name: 被拯救者
            survival_scores: 各选手生存分
            cumulative_tech_scores: 累计技术分
            beta_judge: 评委倾向系数
        """
        # Stage 1: Bottom 2 概率
        # 需要 e 和 s 的生存分都低于其他所有人
        e_score = survival_scores.get(eliminated_name, 0)
        s_score = survival_scores.get(saved_name, 0)
        
        other_scores = [score for name, score in survival_scores.items() 
                       if name not in [eliminated_name, saved_name]]
        
        # 两人都逊于其他人的概率
        bottom2_prob = 1.0
        for other in other_scores:
            # 两人中的高者仍低于该选手
            max_bottom = max(e_score, s_score)
            bottom2_prob *= self.sigmoid(self.alpha * (other - max_bottom))
        
        # Stage 2: 评委选择拯救 s 而不是 e
        tech_s = cumulative_tech_scores.get(saved_name, 0)
        tech_e = cumulative_tech_scores.get(eliminated_name, 0)
        save_prob = self.sigmoid(beta_judge * (tech_s - tech_e))
        
        return bottom2_prob * save_prob


def get_rule_description(season: int) -> str:
    """获取赛季规则描述"""
    if season <= 2:
        return f"Season {season}: Rank Rule (评审排名 + 投票排名)"
    elif season <= 27:
        return f"Season {season}: Percentage Rule (评审百分比 + 投票百分比)"
    else:
        return f"Season {season}: Rank Rule with Judge Save (排名制 + 评委拯救)"


if __name__ == '__main__':
    # 测试规则模块
    rules = CompetitionRules()
    calc = LikelihoodCalculator()
    
    # 测试 Percentage Rule
    print("=== Percentage Rule Test ===")
    judge_scores = {'A': 30, 'B': 25, 'C': 20}
    vote_shares = {'A': 0.4, 'B': 0.35, 'C': 0.25}
    
    results, _ = rules.compute_survival_scores(10, judge_scores, vote_shares)
    for r in results:
        print(f"{r.contestant_name}: S={r.survival_score:.4f}")
    
    # 测试淘汰似然
    print("\n=== Elimination Likelihood Test ===")
    scores = {r.contestant_name: r.survival_score for r in results}
    eliminated = 'C'
    survivors = ['A', 'B']
    
    likelihood = calc.compute_base_elimination_likelihood(
        scores[eliminated],
        [scores[s] for s in survivors]
    )
    print(f"Likelihood of {eliminated} being eliminated: {likelihood:.4f}")
