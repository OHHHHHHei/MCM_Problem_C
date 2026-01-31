import itertools
import pandas as pd
import numpy as np
import time
import concurrent.futures
import os
import csv
from typing import Dict, List

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import core modules
from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse, ModelParams
from core.analysis import ResultAnalyzer

def evaluate_params_worker_robust(params_dict: dict, csv_path: str) -> dict:
    """
    鲁棒的评估工人工 - 支持Early Stopping/Pruning
    """
    try:
        # 1. 初始化
        dp = DataProcessor(csv_path)
        all_seasons = dp.get_seasons()
        
        # 2. 构建参数
        current_params = ModelParams(
            n_particles=400,  # 增加粒子数以获得更稳定的结果
            rho=params_dict.get('rho', 0.6),
            gamma=params_dict.get('gamma', 0.5),
            delta=params_dict.get('delta', 1.5),  # 新增 delta
            alpha_discriminant=params_dict.get('alpha', 5.0),
            kappa=params_dict.get('kappa', 0.3),
            beta_judge=params_dict.get('beta_judge', 2.0)
        )
        
        model = SMCInverse(dp, current_params)
        
        # 3. 手动运行赛季以支持 Early Stopping
        # 策略: 先跑前 10 个赛季。如果 Hit Rate 极差，直接剪枝
        early_stop_seasons = all_seasons[:10]
        results = {}
        
        # Phase 1: 筛选阶段
        for season in early_stop_seasons:
            results[season] = model.run_season(season, verbose=False)
            
        # 检查是否剪枝
        analyzer_early = ResultAnalyzer(model, results)
        early_metrics = analyzer_early.compute_overall_hit_rate()
        
        # 如果前10个赛季 Top-3 命中率低于 40%，大概率是差参数，直接放弃
        if early_metrics['overall_hit_rate_top3'] < 0.40:
            return {
                'params': params_dict,
                'status': 'pruned',
                'top3_hit': early_metrics['overall_hit_rate_top3'],
                'weighted_score': 0.0, # 惩罚低分
                'seasons_run': 10
            }
            
        # Phase 2: 全量阶段
        remaining_seasons = all_seasons[10:]
        for season in remaining_seasons:
            results[season] = model.run_season(season, verbose=False)
            
        # 4. 计算最终指标
        analyzer = ResultAnalyzer(model, results)
        metrics = analyzer.compute_overall_hit_rate()
        consistency = analyzer.compute_consistency_metrics()
        
        # 返回完整结果
        result = {
            'rho': params_dict.get('rho'),
            'gamma': params_dict.get('gamma'),
            'delta': params_dict.get('delta'),
            'alpha': params_dict.get('alpha'),
            'kappa': params_dict.get('kappa'),
            'beta_judge': params_dict.get('beta_judge'),
            'top1_hit': metrics['overall_hit_rate_top1'],
            'top3_hit': metrics['overall_hit_rate_top3'],
            'posterior_consistency': consistency['overall_posterior_consistency'],
            'map_match_rate': consistency['overall_map_match_rate'],
            'seasons_run': len(all_seasons),
            'status': 'completed'
        }
        
        # 计算加权得分
        result['weighted_score'] = (
            result['top3_hit'] * 0.6 + 
            result['map_match_rate'] * 0.2 + 
            result['posterior_consistency'] * 0.2
        )
        
        return result
        
    except Exception as e:
        return {'error': str(e), 'params': params_dict, 'status': 'error'}

def optimize_overnight(csv_path=None):
    if csv_path is None:
        csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2026_MCM_Problem_C_Data.csv'))
    """
    通宵运行优化器 (Overnight Optimizer)
    """
    start_time = time.time()
    output_file = 'output/optimization_overnight_results.csv'
    
    # 1. 定义精细的搜索网格
    # 预计 ~5300 组合
    param_grid = {
        'rho': [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], # 10值
        'gamma': [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],             # 8值
        'delta': [1.2, 1.5, 1.8],                                      # 3值 (新增)
        'alpha': [3.0, 5.0, 7.0],                                      # 3值
        'kappa': [0.2, 0.3, 0.4],                                      # 3值
        'beta_judge': [2.0, 4.0]                                       # 2值
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combinations = len(combinations)
    
    # 按照之前的 Safe Mode，使用 8 核
    max_workers = min(8, os.cpu_count() or 4)
    
    print(f"\n{'='*60}")
    print(f"STARTING OVERNIGHT OPTIMIZATION RUN")
    print(f"{'='*60}")
    print(f"Workers: {max_workers} (Safe Mode)")
    print(f"Particles: 400")
    print(f"Grid Size: {total_combinations} combinations")
    print(f"Features: Checkpointing, Early Stopping (Pruning @ 10 seasons)")
    print(f"Output: {output_file}")
    print(f"{'='*60}\n")
    
    # 2. 初始化输出文件 (Checkpointing)
    os.makedirs('output', exist_ok=True)
    fieldnames = ['rho', 'gamma', 'delta', 'alpha', 'kappa', 'beta_judge', 
                  'top1_hit', 'top3_hit', 'posterior_consistency', 'map_match_rate', 
                  'weighted_score', 'seasons_run', 'status']
    
    # 如果文件不存在，写入表头；如果存在（断点续传），则追加
    file_exists = os.path.isfile(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
    
    print(f"Log initialized at {output_file}...")
    
    # 3. 并行执行
    results_log = []
    completed_count = 0
    pruned_count = 0
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_params = {
            executor.submit(evaluate_params_worker_robust, params, csv_path): params 
            for params in combinations
        }
        
        for future in concurrent.futures.as_completed(future_to_params):
            try:
                result = future.result()
                completed_count += 1
                
                if result.get('status') == 'error':
                    print(f"[{completed_count}/{total_combinations}] Error: {result.get('error')}")
                    continue
                
                # Pruned 结果也记录，但标记
                if result.get('status') == 'pruned':
                    pruned_count += 1
                    # 也可以选择不写入文件以节省空间，这里这为了分析方便还是简略记录
                    # 补全缺失字段以便写入CSV
                    for k in fieldnames:
                        if k not in result:
                            result[k] = 0 # 填充默认值
                    result['rho'] = result['params']['rho']
                    result['gamma'] = result['params']['gamma']
                    # ... 其他参数填充有点麻烦，简化处理：
                    # 只记录 completed 的结果到 CSV，pruned 的只在控制台显示数量
                    if completed_count % 50 == 0:
                        print(f"[{completed_count}/{total_combinations}] Progress update... (Pruned: {pruned_count})")
                    continue
                
                # 记录成功结果
                results_log.append(result)
                
                # Checkpointing: 立即写入 CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    # 过滤掉不需要的键
                    clean_result = {k: result.get(k) for k in fieldnames}
                    writer.writerow(clean_result)
                
                # 实时高分播报
                if result['top3_hit'] > 0.79: # 之前最高是 78.86%
                     print(f"*** NEW HIGH SCORE *** [{completed_count}/{total_combinations}] "
                          f"Top-3: {result['top3_hit']:.2%} | Weighted: {result['weighted_score']:.4f} "
                          f"| Params: rho={result['rho']}, gamma={result['gamma']}, delta={result['delta']}...")

            except Exception as exc:
                print(f"System Error: {exc}")
                
    print("\n" + "="*60)
    print("Overnight Run Complete!")
    print(f"Total Time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Total Completed: {len(results_log)}")
    print(f"Total Pruned: {pruned_count}")
    print("="*60)

if __name__ == '__main__':
    optimize_overnight()
