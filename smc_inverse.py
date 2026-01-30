"""
SMC-Inverse: 序贯蒙特卡洛粒子滤波系统
用于从淘汰数据反演隐藏的粉丝投票份额
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy
import warnings

from data_processor import DataProcessor, Contestant, EliminationEvent
from competition_rules import CompetitionRules, LikelihoodCalculator


@dataclass
class ParticleState:
    """单个粒子的状态"""
    mu: Dict[str, float] = field(default_factory=dict)  # 长期基准 {name: mu}
    x: Dict[str, float] = field(default_factory=dict)   # 短期动量 {name: x}
    weight: float = 1.0
    weight: float = 1.0
    accumulated_shares: Dict[str, float] = field(default_factory=dict)  # 累积投票份额 (Percentage Rule)
    accumulated_vote_ranks: Dict[str, float] = field(default_factory=dict)  # 累积投票排名分 (Rank Rule)
    
    def copy(self) -> 'ParticleState':
        return ParticleState(
            mu=self.mu.copy(),
            x=self.x.copy(),
            weight=self.weight,
            accumulated_shares=self.accumulated_shares.copy(),
            accumulated_vote_ranks=self.accumulated_vote_ranks.copy()
        )


@dataclass
class ModelParams:
    """模型参数"""
    # 状态演化参数
    kappa: float = 0.3          # 表现冲击系数
    delta: float = 1.5          # 冲击阈值 (标准差倍数)
    sigma_mu: float = 0.1       # 长期基准噪声
    rho: float = 0.6            # 记忆衰减系数
    gamma: float = 0.5          # 评审引导系数
    sigma_x: float = 0.2        # 短期动量噪声
    
    # 初始化参数
    sigma_0: float = 0.5        # 初始人气基准标准差
    
    # 规则参数
    xi_perturbation: float = 1e-4
    lambda_decay: float = 0.9
    beta_judge: float = 2.0
    
    # 似然参数
    alpha_discriminant: float = 5.0
    
    # SMC参数
    n_particles: int = 1000
    resample_threshold: float = 0.5  # ESS阈值


class SMCInverse:
    """
    SMC-Inverse 粒子滤波主类
    
    实现基于序贯蒙特卡洛的隐状态反演系统
    """
    
    def __init__(self, 
                 data_processor: DataProcessor,
                 params: Optional[ModelParams] = None):
        """
        初始化 SMC 系统
        
        Args:
            data_processor: 数据处理器
            params: 模型参数
        """
        self.dp = data_processor
        self.params = params or ModelParams()
        
        # 将优化参数传入子模块
        self.rules = CompetitionRules(
            xi_perturbation=self.params.xi_perturbation,
            lambda_decay=self.params.lambda_decay,
            judge_save_beta=self.params.beta_judge  # 传入 beta_judge
        )
        
        self.likelihood_calc = LikelihoodCalculator(
            alpha_discriminant=self.params.alpha_discriminant
        )
        
        # 存储回归系数 (LOSO策略)
        self.beta_coefficients: Dict[int, np.ndarray] = {}
        
        # 粒子集合
        self.particles: List[ParticleState] = []
        
        # 诊断信息
        self.diagnostics: Dict = {
            'ess_history': [],
            'likelihood_history': [],
            'prediction_history': []
        }
    
    def _fit_initial_regression(self, exclude_season: int):
        """
        使用 Expanding Window 策略拟合初始人气回归
        
        只使用 exclude_season 之前的赛季训练，避免时间泄露
        (修正原LOSO策略中使用未来数据的问题)
        """
        X_list = []
        y_list = []
        
        for season in self.dp.get_seasons():
            # 只使用过去赛季的数据 (严格小于当前赛季)
            if season >= exclude_season:
                continue
            
            contestants = self.dp.get_contestants_in_season(season)
            for c in contestants:
                features = self.dp.get_feature_vector(c)
                
                # 使用最终名次作为目标 (越靠前越好)
                # 转换为"受欢迎程度"得分
                n_contestants = len(contestants)
                if n_contestants > 1:
                    popularity = (n_contestants - c.placement) / (n_contestants - 1)
                else:
                    popularity = 0.5
                
                X_list.append(features)
                y_list.append(popularity)
        
        if len(X_list) < 2:
            # 数据不足 (如第1季没有历史数据)，使用先验默认值
            feature_dim = self.dp.get_feature_vector(
                self.dp.get_contestants_in_season(exclude_season)[0]
            ).shape[0]
            self.beta_coefficients[exclude_season] = np.zeros(feature_dim)
            self._intercept = 0.5  # 默认先验均值
            return
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # 简单线性回归 (带正则化)
        try:
            # 添加截距项
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # 岭回归
            lambda_reg = 0.1
            XTX = X_with_intercept.T @ X_with_intercept
            XTy = X_with_intercept.T @ y
            beta = np.linalg.solve(
                XTX + lambda_reg * np.eye(XTX.shape[0]), 
                XTy
            )
            
            self.beta_coefficients[exclude_season] = beta[1:]  # 不含截距
            self._intercept = beta[0]
        except:
            feature_dim = X.shape[1]
            self.beta_coefficients[exclude_season] = np.zeros(feature_dim)
            self._intercept = 0.5
    
    def _initialize_particles(self, season: int, contestants: List[Contestant]):
        """
        初始化粒子集合
        
        对于每个选手，使用回归结果初始化 μ0
        """
        # 确保有回归系数
        if season not in self.beta_coefficients:
            self._fit_initial_regression(season)
        
        beta = self.beta_coefficients[season]
        
        self.particles = []
        for _ in range(self.params.n_particles):
            particle = ParticleState()
            
            for c in contestants:
                features = self.dp.get_feature_vector(c)
                
                # 初始人气基准
                mu_mean = np.dot(beta, features) + self._intercept
                mu_init = mu_mean + np.random.normal(0, self.params.sigma_0)
                
                particle.mu[c.name] = mu_init
                particle.x[c.name] = mu_init  # 初始动量等于基准
            
            particle.weight = 1.0 / self.params.n_particles
            self.particles.append(particle)
    
    def _state_transition(self, 
                         particle: ParticleState,
                         zscore_scores: Dict[str, float],
                         active_names: List[str]) -> ParticleState:
        """
        状态转移方程
        
        式(47): 长期基准演化
        式(58): 短期动量演化
        """
        new_state = particle.copy()
        
        for name in active_names:
            if name not in particle.mu:
                continue
            
            mu_prev = particle.mu[name]
            x_prev = particle.x.get(name, mu_prev)
            j_tilde = zscore_scores.get(name, 0.0)  # 标准化评审分
            
            # 长期基准更新 (式47)
            if abs(j_tilde) > self.params.delta:
                # 有表现冲击
                shock = self.params.kappa * j_tilde
            else:
                shock = 0.0
            
            eta = np.random.normal(0, self.params.sigma_mu)
            mu_new = mu_prev + shock + eta
            
            # 短期动量更新 (式58)
            driving_force = mu_new + self.params.gamma * j_tilde
            epsilon = np.random.normal(0, self.params.sigma_x)
            x_new = self.params.rho * x_prev + (1 - self.params.rho) * driving_force + epsilon
            
            new_state.mu[name] = mu_new
            new_state.x[name] = x_new
        
        return new_state
    
    def _compute_vote_shares(self, 
                            particle: ParticleState,
                            active_names: List[str]) -> Dict[str, float]:
        """
        计算投票份额 (Softmax映射)
        
        式(69): π_i,t = exp(x_i,t) / Σ_k exp(x_k,t)
        """
        x_values = {}
        for name in active_names:
            x_values[name] = particle.x.get(name, 0.0)
        
        # Softmax with numerical stability
        max_x = max(x_values.values()) if x_values else 0.0
        exp_x = {name: np.exp(x - max_x) for name, x in x_values.items()}
        sum_exp = sum(exp_x.values())
        
        if sum_exp == 0:
            sum_exp = 1.0
        
        return {name: exp / sum_exp for name, exp in exp_x.items()}
    
    def _compute_particle_likelihood(self,
                                     particle: ParticleState,
                                     season: int,
                                     week: int,
                                     elimination_event: EliminationEvent,
                                     judge_scores: Dict[str, float],
                                     use_accumulated_shares: bool = False,
                                     override_vote_shares: Optional[Dict[str, float]] = None) -> float:
        """
        计算粒子似然
        
        修正: 
        - 使用集合淘汰似然处理多人淘汰周
        - S28+ 使用完整的 Judge Save 两阶段似然
        - Rank Rule 无淘汰周后使用累积投票份额
        """
        # 获取在场选手 (被淘汰者列表 + 幸存者)
        active_names = elimination_event.eliminated_set + elimination_event.survivors
        
        # 计算当周投票份额
        if override_vote_shares is not None:
             vote_shares = override_vote_shares
        else:
            current_shares = self._compute_vote_shares(particle, active_names)
            
            # 对于 Rank Rule 赛季，如果需要累积份额
            rule_type = self.rules._get_rule_type(season)
            if use_accumulated_shares and rule_type in ['RANK', 'RANK_WITH_SAVE']:
                # 使用累积份额 (当前份额 + 之前累积的份额)
                vote_shares = {}
                for name in active_names:
                    curr = current_shares.get(name, 0)
                    acc = particle.accumulated_shares.get(name, 0)
                    vote_shares[name] = curr + acc
                # 重新归一化
                total = sum(vote_shares.values())
                if total > 0:
                    vote_shares = {k: v / total for k, v in vote_shares.items()}
            else:
                vote_shares = current_shares
        
        # 计算生存分
        survival_scores, bottom_two = self.rules.compute_survival_scores(
            season, judge_scores, vote_shares
        )
        
        score_dict = {s.contestant_name: s.survival_score for s in survival_scores}
        
        rule_type = self.rules._get_rule_type(season)
        
        if rule_type == 'RANK_WITH_SAVE' and bottom_two and len(elimination_event.eliminated_set) == 1:
            # S28+ Judge Save 机制 (仅单人淘汰时适用)
            # 两阶段似然:
            # Stage 1: 被淘汰者在 Bottom 2 中
            # Stage 2: 评委选择拯救另一人 (基于技术分)
            
            eliminated_name = elimination_event.eliminated_set[0]
            
            # 找出谁被拯救了: 应该是 Bottom 2 中的另一人
            # 推断: 下周仍在场的 Bottom 2 成员 = 被拯救者
            # 简化: 假设 Bottom 2 中分数较高者被拯救
            b2_name1, b2_name2 = bottom_two
            
            if eliminated_name in [b2_name1, b2_name2]:
                # 被淘汰者确实在 Bottom 2 中，符合 Judge Save 逻辑
                saved_name = b2_name1 if eliminated_name == b2_name2 else b2_name2
                
                # Stage 1: Bottom 2 概率 (两人分数低于其他人)
                b2_scores = [score_dict.get(b2_name1, 0), score_dict.get(b2_name2, 0)]
                max_b2_score = max(b2_scores)
                
                other_scores = [score_dict.get(name, 0) for name in elimination_event.survivors
                               if name not in [b2_name1, b2_name2]]
                
                bottom2_prob = 1.0
                for other_score in other_scores:
                    # 其他人分数都高于 Bottom 2 中的最高者
                    bottom2_prob *= self.likelihood_calc.sigmoid(
                        self.likelihood_calc.alpha * (other_score - max_b2_score)
                    )
                
                # Stage 2: 评委拯救概率 (基于累计评审分)
                # 评委倾向拯救技术分更高的选手
                tech_saved = judge_scores.get(saved_name, 0)
                tech_elim = judge_scores.get(eliminated_name, 0)
                save_prob = self.likelihood_calc.sigmoid(
                    self.rules.beta_judge * (tech_saved - tech_elim)
                )
                
                likelihood = bottom2_prob * save_prob
            else:
                # 被淘汰者不在该粒子的 Bottom 2 中，极低似然
                likelihood = 1e-10
        else:
            # 非 Judge Save 赛季，或多人淘汰
            # 使用集合淘汰似然 (统一处理单人和多人淘汰)
            eliminated_scores = [score_dict.get(e, 0.0) for e in elimination_event.eliminated_set]
            survivor_scores = [score_dict.get(s, 0.0) for s in elimination_event.survivors]
            
            likelihood = self.likelihood_calc.compute_group_elimination_likelihood(
                eliminated_scores, survivor_scores
            )
        
        return max(likelihood, 1e-300)  # 防止零似然
    
    def _effective_sample_size(self) -> float:
        """计算有效样本量 (ESS)"""
        weights = np.array([p.weight for p in self.particles])
        weights = weights / weights.sum()  # 归一化
        return 1.0 / np.sum(weights ** 2)
    
    def _resample(self):
        """系统重采样"""
        n = len(self.particles)
        weights = np.array([p.weight for p in self.particles])
        weights = weights / weights.sum()
        
        # 系统重采样
        positions = (np.arange(n) + np.random.uniform()) / n
        cumsum = np.cumsum(weights)
        
        indices = []
        i, j = 0, 0
        while i < n:
            if positions[i] < cumsum[j]:
                indices.append(j)
                i += 1
            else:
                j += 1
        
        # 创建新粒子集合
        new_particles = []
        for idx in indices:
            new_particle = self.particles[idx].copy()
            new_particle.weight = 1.0 / n
            new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def _predict_elimination_probabilities(self,
                                           season: int,
                                           week: int,
                                           active_contestants: List[Contestant],
                                           judge_scores: Dict[str, float]) -> Dict[str, float]:
        """
        预测淘汰概率 (先验预测，重采样前)
        """
        elimination_counts = {c.name: 0 for c in active_contestants}
        total_weight = 0.0
        
        for particle in self.particles:
            # 计算投票份额
            active_names = [c.name for c in active_contestants]
            vote_shares = self._compute_vote_shares(particle, active_names)
            
            # 计算生存分
            survival_scores, _ = self.rules.compute_survival_scores(
                season, judge_scores, vote_shares
            )
            
            # 找出最低分者
            if survival_scores:
                min_score = min(s.survival_score for s in survival_scores)
                for s in survival_scores:
                    if s.survival_score == min_score:
                        elimination_counts[s.contestant_name] += particle.weight
                        break
            
            total_weight += particle.weight
        
        if total_weight == 0:
            total_weight = 1.0
        
        return {name: count / total_weight for name, count in elimination_counts.items()}
    
    def run_season(self, season: int, verbose: bool = True) -> Dict:
        """
        运行单赛季的粒子滤波
        
        Returns:
            results: 包含投票份额估计、预测精度等
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"Processing Season {season}")
            print(f"Rule Type: {self.rules._get_rule_type(season)}")
            print(f"{'='*50}")
        
        # 重置规则状态
        self.rules.reset_season()
        
        # 获取淘汰事件
        events = self.dp.get_elimination_events(season)
        if not events:
            if verbose:
                print("No elimination events found for this season.")
            return {}
        
        # 获取首周选手并初始化粒子
        first_week = min(e.week for e in events)
        initial_contestants = self.dp.get_active_contestants(season, first_week)
        
        if not initial_contestants:
            if verbose:
                print("No contestants found for initial week.")
            return {}
        
        self._initialize_particles(season, initial_contestants)
        
        results = {
            'season': season,
            'weekly_estimates': {},
            'predictions': [],
            'hit_rates': [],
            'log_likelihoods': []
        }
        
        # 评委分累加器 (用于无淘汰周的分数滚存)
        judge_score_accumulator = {}
        prev_week = None
        elimination_weeks = {e.week for e in events}
        
        # 找出所有需要处理的周次 (包括无淘汰周)
        all_weeks = set()
        for c in initial_contestants:
            all_weeks.update(c.weekly_scores.keys())
        
        # 创建事件查找表
        events_by_week = {e.week: e for e in events}
        
        # 规则类型
        rule_type = self.rules._get_rule_type(season)
        
        # 遍历每一周 (关键修改: 不再只遍历淘汰周)
        for week in sorted(all_weeks):
            if week < first_week:
                continue
            
            # 获取在场选手
            active = self.dp.get_active_contestants(season, week)
            if not active:
                continue
            
            active_names = [c.name for c in active]
            
            # 获取标准化分数 (用于状态转移)
            zscore_scores = self.dp.compute_zscore_scores(season, week)
            
            # === A. 状态转移 (每周都要执行，无论是否有淘汰) ===
            for i in range(len(self.particles)):
                self.particles[i] = self._state_transition(
                    self.particles[i], zscore_scores, active_names
                )
            
            # === B. 累积逻辑 (Accumulation) ===
            # 根据规则类型执行不同的累积策略
            if rule_type == 'PERCENTAGE':
                # 百分比规则: 累积份额 (Share Accumulation)
                for particle in self.particles:
                    current_shares = self._compute_vote_shares(particle, active_names)
                    for name in active_names:
                        particle.accumulated_shares[name] = (
                            particle.accumulated_shares.get(name, 0) + current_shares.get(name, 0)
                        )
            
            elif rule_type in ['RANK', 'RANK_WITH_SAVE']:
                # 排名规则: 累积排名分 (Points Accumulation)
                # "Add those ranking points together"
                for particle in self.particles:
                    current_shares = self._compute_vote_shares(particle, active_names)
                    
                    # 计算当周排名 (1为最高分/最高份额)
                    # 注意: DWTS中Rank Point通常是排名本身(1st=1pt)，目标是分数越低越好
                    # 或者反过来，但我们的competition_rules里用的是 Rank(1..N)，Rank 1 最好。
                    # 加微小随机噪声防止平局相同处理
                    noise = {n: np.random.uniform(0, 1e-9) for n in active_names}
                    sorted_by_share = sorted(
                        active_names, 
                        key=lambda n: -(current_shares.get(n, 0) + noise[n])
                    )
                    
                    current_ranks = {name: i+1 for i, name in enumerate(sorted_by_share)}
                    
                    for name in active_names:
                        particle.accumulated_vote_ranks[name] = (
                            particle.accumulated_vote_ranks.get(name, 0) + current_ranks.get(name, 0)
                        )
            
            # === C. 检查本周是否有淘汰事件 ===
            event = events_by_week.get(week, None)
            
            if event:
                # ============ 淘汰周: 观测更新 ============
                
                # 获取当周原始分数
                current_week_scores = {c.name: self.dp.get_weekly_total_score(c, week) for c in active}
                
                # 评委分累加: 回溯检查连续的无淘汰周
                accumulated_scores = current_week_scores.copy()
                check_week = week - 1
                while check_week >= 1 and check_week not in elimination_weeks:
                    for c in active:
                        if check_week in c.weekly_scores:
                            prev_score = self.dp.get_weekly_total_score(c, check_week)
                            accumulated_scores[c.name] = accumulated_scores.get(c.name, 0) + prev_score
                    check_week -= 1
                
                judge_scores = accumulated_scores
                
                # 先验预测 (重采样前)
                # 注意: _predict_elimination_probabilities 也需要更新以支持累积
                # 但由于该函数只是辅助用的，并不影响核心滤波，暂时简单跳过或仅传入当前
                # 真正的优化应该也传入 accumulated_xxx
                pred_probs = self._predict_elimination_probabilities(
                    season, week, active, judge_scores
                )
                
                # 准备投票数据供似然计算
                # 对于 Rank Rule, 传入累积排名分
                # 对于 Percentage Rule, 传入累积份额
                vote_metric_for_likelihood = {}
                
                if rule_type == 'PERCENTAGE':
                    # 使用累积份额
                    for particle in self.particles:
                        # 临时存储到 particle.vote_shares_buffer 以传给 likelihood
                        # 或者修改 _compute_particle_likelihood 接口
                        # 这里我们在 _compute_particle_likelihood 内部处理，只需要传入 flag
                        pass
                
                # 计算似然并更新权重
                log_likelihoods = []
                for particle in self.particles:
                    # 获取该粒子用于判定的"投票数据"
                    if rule_type == 'PERCENTAGE':
                        # 份额累积
                        p_shares = self._compute_vote_shares(particle, active_names)
                        vote_data = {
                            n: p_shares.get(n, 0) + particle.accumulated_shares.get(n, 0)
                            for n in active_names
                        }
                        # 归一化
                        total_s = sum(vote_data.values())
                        if total_s > 0:
                            vote_data = {k: v/total_s for k, v in vote_data.items()}
                        
                        use_rank_points = False
                        
                    elif rule_type in ['RANK', 'RANK_WITH_SAVE']:
                        # 排名分累积 (直接使用累积后的 rank points)
                        # 注意: 每一周增加一次排名分，所以 accumulated_vote_ranks 已经包含了本周 + 历史
                        # 我们需要将其作为 "vote_shares" 传入 compute_survival_scores
                        # 但 compute_survival_scores 会对传入的数据再次求 Rank
                        # 这正是我们在思考中提到的问题: Rank(Sum(Points)) vs Sum(Points)
                        # 用户指出: "Add those ranking points together... total points lowest eliminated"
                        # 所以最终 Survival Score = JudgeRankSum + VoteRankSum
                        # 我们传入 VoteRankSum 给 compute_survival_scores, 它会再次 Rank 吗？
                        # 会的。 compute_rank_rule_score 内部: vote_ranks = rank(fan_vote_shares)
                        # 如果我们传入 VoteRankSum，它会算出 Rank(VoteRankSum)。
                        # 这与 "Total Points Lowest" 是等价的吗？
                        # 是的。因为 Points 越小越好 (1st=1pt)。Sum 越小越好。
                        # 对 Sum 求 Rank (1..N)，依然是 Sum 最小者 Rank=1。
                        # 所以我们可以传入 Accumulated Rank Points 作为 "伪造的 Share"，
                        # 但要注意: Share 是越大越好，Rank Points 是越小越好。
                        # 所以我们需要传入 -1 * Accumulated Rank Points，或者 1/Points
                        # 让 compute_survival_scores 的排序逻辑 (大者排前) 正确工作。
                        # compute_rank_rule_score 内部: sorted(..., key=-x[1]) -> 大者排前
                        # 我们希望 Points 小者排前 -> Points 小者 = "伪Share" 大者
                        # 所以传入 1.0 / AccumulatedPoints 是合理的代理
                        
                        current_vote_ranks = particle.accumulated_vote_ranks
                        vote_data = {
                            n: 1.0 / (current_vote_ranks.get(n, 999) + 1e-6) 
                            for n in active_names
                        }
                        # 仅作为排序依据，不需要归一化
                        use_rank_points = True
                        
                    else:
                        vote_data = self._compute_vote_shares(particle, active_names)
                        use_rank_points = False

                    # 调用似然计算
                    # 注意: 我们直接传入处理好的 vote_data，不再在内部进行累积处理
                    ll = self._compute_particle_likelihood(
                        particle, season, week, event, judge_scores, 
                        use_accumulated_shares=False, # We already handled accumulation
                        override_vote_shares=vote_data
                    )
                    particle.weight *= ll
                    log_likelihoods.append(np.log(ll + 1e-300))
                
                results['log_likelihoods'].append({
                    'week': week,
                    'eliminated': event.eliminated_set,
                    'mean_log_likelihood': np.mean(log_likelihoods),
                    'is_multi_elimination': event.is_multi_elimination
                })
                
                # 归一化权重
                total_weight = sum(p.weight for p in self.particles)
                if total_weight > 0:
                    for p in self.particles:
                        p.weight /= total_weight
                
                # 记录ESS
                ess = self._effective_sample_size()
                self.diagnostics['ess_history'].append({
                    'season': season, 'week': week, 'ess': ess
                })
                
                # 检验预测
                sorted_preds = sorted(pred_probs.items(), key=lambda x: -x[1])
                top_1_names = [name for name, _ in sorted_preds[:1]]
                top_3_names = [name for name, _ in sorted_preds[:3]]
                
                for elim_name in event.eliminated_set:
                    hit_top1 = elim_name in top_1_names
                    hit_top3 = elim_name in top_3_names
                    results['predictions'].append({
                        'week': week,
                        'eliminated': elim_name,
                        'predicted_probs': pred_probs,
                        'hit_top1': hit_top1,
                        'hit_top3': hit_top3
                    })
                
                # 重采样
                if ess < self.params.resample_threshold * self.params.n_particles:
                    self._resample()
                    if verbose:
                        elim_str = ', '.join(event.eliminated_set)
                        print(f"  Week {week}: ESS={ess:.1f}, Resampled (eliminated: {elim_str})")
                elif verbose:
                    elim_str = ', '.join(event.eliminated_set)
                    print(f"  Week {week}: ESS={ess:.1f} (eliminated: {elim_str})")
                
                # 估计投票份额
                estimates = self._estimate_vote_shares(active_names)
                results['weekly_estimates'][week] = estimates
                
                # ============ 新增: 一致性验证 (Consistency Metrics) ============
                # 1. 后验一致概率
                consistent_weight = 0.0
                total_weight_check = 0.0
                
                for particle in self.particles:
                    # 重新构建 vote_data (同上)
                    if rule_type == 'PERCENTAGE':
                        p_shares = self._compute_vote_shares(particle, active_names)
                        vote_data = {
                            n: p_shares.get(n, 0) + particle.accumulated_shares.get(n, 0)
                            for n in active_names
                        }
                        s_comb = sum(vote_data.values())
                        if s_comb > 0: vote_data = {k: v/s_comb for k, v in vote_data.items()}
                        
                    elif rule_type in ['RANK', 'RANK_WITH_SAVE']:
                        vote_data = {
                            n: 1.0 / (particle.accumulated_vote_ranks.get(n, 999) + 1e-6)
                            for n in active_names
                        }
                    else:
                        vote_data = self._compute_vote_shares(particle, active_names)
                        
                    simulated_eliminated = self.rules.simulate_elimination_outcome(
                        season, judge_scores, vote_data
                    )
                    
                    if simulated_eliminated in event.eliminated_set:
                        consistent_weight += particle.weight
                    
                    total_weight_check += particle.weight
                
                posterior_consistency = consistent_weight / total_weight_check if total_weight_check > 0 else 0.0
                
                # 2. MAP 一致性
                # 对于 Point Accumulation, 我们不能简单的平均 "1/Rank"
                # 应该平均 "Rank Points" 本身，然后再取倒数
                if rule_type in ['RANK', 'RANK_WITH_SAVE']:
                    avg_rank_points = {name: 0.0 for name in active_names}
                    for p in self.particles:
                        for n in active_names:
                            avg_rank_points[n] += p.accumulated_vote_ranks.get(n, 0) * p.weight
                    
                    # 构造用于模拟的输入
                    map_shares_comb = {k: 1.0/(v+1e-6) for k, v in avg_rank_points.items()}
                
                elif rule_type == 'PERCENTAGE':
                     # Percentage 平均累积份额
                    avg_shares = {name: 0.0 for name in active_names}
                    for p in self.particles:
                        p_shares = self._compute_vote_shares(p, active_names) # 当周
                        for n in active_names:
                            # 当周 + 历史
                            val = p_shares.get(n,0) + p.accumulated_shares.get(n,0)
                            avg_shares[n] += val * p.weight
                     
                    map_shares_comb = avg_shares
                    # 归一化
                    s = sum(map_shares_comb.values())
                    if s>0: map_shares_comb = {k: v/s for k, v in map_shares_comb.items()}
                else:
                    # 普通周 (非累积)
                    # ... (已有逻辑)
                     mean_shares = {name: estimates[name]['mean'] for name in active_names}
                     map_shares_comb = mean_shares
                
                map_eliminated = self.rules.simulate_elimination_outcome(
                    season, judge_scores, map_shares_comb
                )
                
                map_consistent = map_eliminated in event.eliminated_set
                
                # 记录结果
                results.setdefault('consistency_metrics', []).append({
                    'week': week,
                    'posterior_consistency': posterior_consistency,
                    'map_consistent': map_consistent,
                    'eliminated': list(event.eliminated_set),
                    'map_predicted': map_eliminated
                })
                
                # 更新淘汰状态
                self.rules.update_elimination_status(True)
                
                # 淘汰周后重置累积
                for p in self.particles:
                    p.accumulated_shares = {}
                    p.accumulated_vote_ranks = {}
            
            else:
                # ============ 无淘汰周: 仅状态演化 ============
                # 状态转移已在上面完成
                # 份额累积已在上面完成 (Rank Rule)
                # 不计算似然，不重采样
                if verbose:
                    print(f"  Week {week}: No elimination (state transition only)")
        
        # 计算整体命中率 (Top-1 和 Top-3)
        if results['predictions']:
            hit_count_top1 = sum(1 for p in results['predictions'] if p['hit_top1'])
            hit_count_top3 = sum(1 for p in results['predictions'] if p['hit_top3'])
            results['hit_rate_top1'] = hit_count_top1 / len(results['predictions'])
            results['hit_rate_top3'] = hit_count_top3 / len(results['predictions'])
        else:
            results['hit_rate_top1'] = 0.0
            results['hit_rate_top3'] = 0.0
        
        if verbose:
            print(f"\nSeason {season} Summary:")
            print(f"  Total elimination events: {len(events)}")
            print(f"  Top-1 Hit Rate: {results['hit_rate_top1']:.2%}")
            print(f"  Top-3 Hit Rate: {results['hit_rate_top3']:.2%}")
            
            # 打印新的一致性指标
            if 'consistency_metrics' in results:
                avg_post = np.mean([m['posterior_consistency'] for m in results['consistency_metrics']])
                avg_map = np.mean([1.0 if m['map_consistent'] else 0.0 for m in results['consistency_metrics']])
                print(f"  Posterior Consistency: {avg_post:.2%}")
                print(f"  MAP Match Rate: {avg_map:.2%}")
        
        return results
    
    def _estimate_vote_shares(self, active_names: List[str]) -> Dict[str, Dict]:
        """
        估计投票份额 (后验分布)
        
        Returns:
            {name: {'mean': float, 'std': float, 'ci_low': float, 'ci_high': float}}
        """
        estimates = {name: {'shares': []} for name in active_names}
        
        for particle in self.particles:
            shares = self._compute_vote_shares(particle, active_names)
            for name, share in shares.items():
                estimates[name]['shares'].append(share * particle.weight)
        
        # 计算统计量
        for name in estimates:
            shares = np.array(estimates[name]['shares'])
            total_weight = sum(p.weight for p in self.particles)
            
            if total_weight > 0:
                mean = np.sum(shares) / total_weight
            else:
                mean = 1.0 / len(active_names)
            
            # 近似标准差
            std = np.std(shares) if len(shares) > 1 else 0.0
            
            estimates[name] = {
                'mean': mean,
                'std': std,
                'ci_low': max(0, mean - 1.96 * std),
                'ci_high': min(1, mean + 1.96 * std)
            }
        
        return estimates
    
    def run_all_seasons(self, 
                        seasons: Optional[List[int]] = None,
                        verbose: bool = True) -> Dict[int, Dict]:
        """
        运行所有赛季
        
        Args:
            seasons: 指定赛季列表，None则运行所有
            verbose: 是否打印详细信息
            
        Returns:
            {season: results}
        """
        if seasons is None:
            seasons = self.dp.get_seasons()
        
        all_results = {}
        for season in seasons:
            try:
                results = self.run_season(season, verbose)
                all_results[season] = results
            except Exception as e:
                warnings.warn(f"Error processing season {season}: {e}")
                continue
        
        return all_results


def create_model(csv_path: str = '2026_MCM_Problem_C_Data.csv',
                 n_particles: int = 1000) -> SMCInverse:
    """便捷函数：创建模型"""
    dp = DataProcessor(csv_path)
    params = ModelParams(n_particles=n_particles)
    return SMCInverse(dp, params)


if __name__ == '__main__':
    # 测试模型
    print("Creating SMC-Inverse model...")
    model = create_model(n_particles=500)
    
    # 测试单赛季
    print("\nTesting on Season 27...")
    results = model.run_season(27)
    
    if results:
        print(f"\nFinal Hit Rate: {results.get('hit_rate_top3', 0):.2%}")
