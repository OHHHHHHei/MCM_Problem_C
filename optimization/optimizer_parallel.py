import itertools
import pandas as pd
import numpy as np
import time
import concurrent.futures
import os
from copy import deepcopy

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core modules
from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams
from core.analysis import ResultAnalyzer

# 定义全局 worker 函数，必须在顶层以便 pickle
def evaluate_params_worker(params_dict: dict, csv_path: str, season_list: list = None) -> dict:
    """
    单个参数组合的评估任务，由子进程执行
    """
    try:
        # 在子进程中重新加载数据，避免跨进程共享大对象带来的开销/问题
        # DataProcessor 加载很快
        dp = DataProcessor(csv_path)
        
        # 如果未指定赛季，使用全部
        test_seasons = season_list if season_list else dp.get_seasons()
        
        # 构建模型参数
        # 注意：这里我们优化 5 个关键参数
        current_params = ModelParams(
            n_particles=200,  # 搜索时使用较少粒子平衡速度
            rho=params_dict.get('rho', 0.6),
            gamma=params_dict.get('gamma', 0.5),
            alpha_discriminant=params_dict.get('alpha', 5.0),
            kappa=params_dict.get('kappa', 0.3),
            beta_judge=params_dict.get('beta_judge', 2.0)
        )
        
        # 初始化模型
        model = SMCInverse(dp, current_params)
        
        # 运行模拟 (关闭详细输出)
        results = model.run_all_seasons(seasons=test_seasons, verbose=False)
        
        # 计算指标
        analyzer = ResultAnalyzer(model, results)
        metrics = analyzer.compute_overall_hit_rate()
        consistency = analyzer.compute_consistency_metrics()
        
        # 返回结果字典
        result = {
            'rho': params_dict.get('rho'),
            'gamma': params_dict.get('gamma'),
            'alpha': params_dict.get('alpha'),
            'kappa': params_dict.get('kappa'),
            'beta_judge': params_dict.get('beta_judge'),
            'top1_hit': metrics['overall_hit_rate_top1'],
            'top3_hit': metrics['overall_hit_rate_top3'],
            'posterior_consistency': consistency['overall_posterior_consistency'],
            'map_match_rate': consistency['overall_map_match_rate']
        }
        
        # 计算加权得分用于排序
        # 权重: Top-3 Hit (0.5) + MAP Match (0.3) + Posterior Consistency (0.2)
        result['weighted_score'] = (
            result['top3_hit'] * 0.5 + 
            result['map_match_rate'] * 0.3 + 
            result['posterior_consistency'] * 0.2
        )
        
        return result
        
    except Exception as e:
        # 捕获异常防止子进程崩溃导致主进程卡死
        return {'error': str(e), 'params': params_dict}

def optimize_hyperparameters_parallel(csv_path=None):
    if csv_path is None:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2026_MCM_Problem_C_Data.csv'))
    """
    并行网格搜索优化超参数 (Aggressive Mode)
    """
    start_time = time.time()
    
    # 1. 定义激进的参数网格
    # 共约 450 种组合
    param_grid = {
        'rho': [0.5, 0.6, 0.7, 0.8, 0.9],            # 记忆衰减 (5个值)
        'gamma': [0.0, 0.2, 0.4, 0.6, 0.8],          # 评审引导 (5个值)
        'alpha': [3.0, 5.0, 7.0],                    # 判别力 (3个值)
        'kappa': [0.2, 0.3, 0.4],                    # 冲击系数 (3个值)
        'beta_judge': [2.0, 4.0]                     # 评委拯救系数 (2个值)
    }
    
    # 生成所有组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combinations = len(combinations)
    # 根据CPU核心数确定并发数
    # Windows下为了稳定性，限制最大并发数为8
    # 虽然i9-13980HX有32线程，但过度并发会导致进程初始化失败
    max_workers = min(10, os.cpu_count() or 4)
    
    print(f"\n{'='*50}")
    print(f"STARTING AGGRESSIVE PARALLEL OPTIMIZATION")
    print(f"{'='*50}")
    print(f"Hardware: Detected {os.cpu_count()} logical cores.")
    print(f"Strategy: Parallel Process Pool with {max_workers} workers (Safe Mode).")
    print(f"Search Space: {total_combinations} combinations.")
    print(f"Estimated Time: ~20-30 minutes.")
    print(f"{'='*50}\n")
    
    results_log = []
    completed_count = 0
    
    # 使用 ProcessPoolExecutor 进行并行计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_params = {
            executor.submit(evaluate_params_worker, params, csv_path): params 
            for params in combinations
        }
        
        # 异步获取结果
        for future in concurrent.futures.as_completed(future_to_params):
            try:
                result = future.result()
                completed_count += 1
                
                if 'error' in result:
                    print(f"[{completed_count}/{total_combinations}] Error: {result['error']}")
                else:
                    results_log.append(result)
                    # 简略进度条
                    if completed_count % 10 == 0 or result['top3_hit'] > 0.78:
                        score_fmt = f"{result['weighted_score']:.4f}"
                        print(f"[{completed_count}/{total_combinations}] "
                              f"Top-3: {result['top3_hit']:.2%} | "
                              f"MAP: {result['map_match_rate']:.2%} | "
                              f"Params: rho={result['rho']}, gamma={result['gamma']}, alpha={result['alpha']}...")
                        
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    # 5. 分析结果
    if not results_log:
        print("No results collected.")
        return

    df = pd.DataFrame(results_log)
    df = df.sort_values(by='weighted_score', ascending=False)
    
    best_run = df.iloc[0]
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Time elapsed: {time.time() - start_time:.2f}s")
    print("-" * 30)
    print("Best Parameters found:")
    print(f"  rho (Memory): {best_run['rho']}")
    print(f"  gamma (Judge): {best_run['gamma']}")
    print(f"  alpha (Discriminant): {best_run['alpha']}")
    print(f"  kappa (Impact): {best_run['kappa']}")
    print(f"  beta_judge (Save): {best_run['beta_judge']}")
    print("-" * 30)
    print(f"Best Top-3 Hit Rate: {best_run['top3_hit']:.2%}")
    print(f"Best MAP Match Rate: {best_run['map_match_rate']:.2%}")
    print(f"Best Consistency:    {best_run['posterior_consistency']:.2%}")
    print("="*50)
    
    # 保存结果
    output_path = 'output/optimization_aggressive_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Full results saved to {output_path}")

if __name__ == '__main__':
    # Windows下使用multiprocessing必须放在if __name__ == '__main__':之下
    optimize_hyperparameters_parallel()
