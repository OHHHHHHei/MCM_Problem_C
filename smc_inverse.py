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
    
    def copy(self) -> 'ParticleState':
        return ParticleState(
            mu=self.mu.copy(),
            x=self.x.copy(),
            weight=self.weight
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
        
        self.rules = CompetitionRules(
            xi_perturbation=self.params.xi_perturbation,
            lambda_decay=self.params.lambda_decay,
            judge_save_beta=self.params.beta_judge
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
        使用 LOSO 策略拟合初始人气回归
        
        排除 exclude_season 季的数据，使用其他所有赛季训练
        """
        X_list = []
        y_list = []
        
        for season in self.dp.get_seasons():
            if season == exclude_season:
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
            # 数据不足，使用零系数
            feature_dim = self.dp.get_feature_vector(
                self.dp.get_contestants_in_season(exclude_season)[0]
            ).shape[0]
            self.beta_coefficients[exclude_season] = np.zeros(feature_dim)
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
                                     judge_scores: Dict[str, float]) -> float:
        """
        计算粒子似然
        """
        # 获取在场选手
        active_names = [elimination_event.eliminated] + elimination_event.survivors
        
        # 计算投票份额
        vote_shares = self._compute_vote_shares(particle, active_names)
        
        # 计算生存分
        survival_scores, bottom_two = self.rules.compute_survival_scores(
            season, judge_scores, vote_shares
        )
        
        score_dict = {s.contestant_name: s.survival_score for s in survival_scores}
        
        # 获取被淘汰者和幸存者的分数
        eliminated_score = score_dict.get(elimination_event.eliminated, 0.0)
        survivor_scores = [score_dict.get(s, 0.0) for s in elimination_event.survivors]
        
        rule_type = self.rules._get_rule_type(season)
        
        if rule_type == 'RANK_WITH_SAVE' and bottom_two:
            # S28+ Judge Save 机制
            # 需要两阶段似然
            # 暂时简化：使用基础淘汰似然
            likelihood = self.likelihood_calc.compute_base_elimination_likelihood(
                eliminated_score, survivor_scores
            )
        else:
            # 基础淘汰似然
            likelihood = self.likelihood_calc.compute_base_elimination_likelihood(
                eliminated_score, survivor_scores
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
        
        # 按周处理淘汰事件
        events_by_week = {}
        for e in events:
            if e.week not in events_by_week:
                events_by_week[e.week] = []
            events_by_week[e.week].append(e)
        
        for week in sorted(events_by_week.keys()):
            week_events = events_by_week[week]
            
            # 获取在场选手和分数
            active = self.dp.get_active_contestants(season, week)
            if not active:
                continue
            
            active_names = [c.name for c in active]
            
            # 获取标准化分数
            zscore_scores = self.dp.compute_zscore_scores(season, week)
            
            # 获取原始分数
            judge_scores = {c.name: self.dp.get_weekly_total_score(c, week) for c in active}
            
            # 状态转移
            for i in range(len(self.particles)):
                self.particles[i] = self._state_transition(
                    self.particles[i], zscore_scores, active_names
                )
            
            # 先验预测 (重采样前)
            pred_probs = self._predict_elimination_probabilities(
                season, week, active, judge_scores
            )
            
            # 处理每个淘汰事件
            for event in week_events:
                # 计算似然并更新权重
                log_likelihoods = []
                for particle in self.particles:
                    ll = self._compute_particle_likelihood(
                        particle, season, week, event, judge_scores
                    )
                    particle.weight *= ll
                    log_likelihoods.append(np.log(ll + 1e-300))
                
                results['log_likelihoods'].append({
                    'week': week,
                    'eliminated': event.eliminated,
                    'mean_log_likelihood': np.mean(log_likelihoods)
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
            for event in week_events:
                # 计算命中率 (被淘汰者是否在预测的危险区)
                sorted_preds = sorted(pred_probs.items(), key=lambda x: -x[1])
                top_3_names = [name for name, _ in sorted_preds[:3]]
                
                hit = event.eliminated in top_3_names
                results['predictions'].append({
                    'week': week,
                    'eliminated': event.eliminated,
                    'predicted_probs': pred_probs,
                    'hit_top3': hit
                })
            
            # 重采样
            if ess < self.params.resample_threshold * self.params.n_particles:
                self._resample()
                if verbose:
                    print(f"  Week {week}: ESS={ess:.1f}, Resampled")
            elif verbose:
                print(f"  Week {week}: ESS={ess:.1f}")
            
            # 估计投票份额
            estimates = self._estimate_vote_shares(active_names)
            results['weekly_estimates'][week] = estimates
            
            # 更新淘汰状态
            self.rules.update_elimination_status(len(week_events) > 0)
        
        # 计算整体命中率
        if results['predictions']:
            hit_count = sum(1 for p in results['predictions'] if p['hit_top3'])
            results['hit_rate_top3'] = hit_count / len(results['predictions'])
        else:
            results['hit_rate_top3'] = 0.0
        
        if verbose:
            print(f"\nSeason {season} Summary:")
            print(f"  Total elimination events: {len(events)}")
            print(f"  Top-3 Hit Rate: {results['hit_rate_top3']:.2%}")
        
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
