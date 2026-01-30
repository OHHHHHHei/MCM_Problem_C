import itertools
import pandas as pd
import numpy as np
from smc_inverse import SMCInverse, ModelParams
from data_processor import DataProcessor
from analysis import ResultAnalyzer
import time

def optimize_hyperparameters(csv_path='2026_MCM_Problem_C_Data.csv'):
    """
    网格搜索优化超参数
    """
    start_time = time.time()
    
    # 1. 定义要搜索的参数网格
    # 缩小范围以节省时间，基于前期经验
    param_grid = {
        'rho': [0.5, 0.65, 0.8],          # 记忆衰减 (覆盖 低-中-高)
        'gamma': [0.3, 0.5, 0.7],         # 评审引导
        'alpha_discriminant': [5.0]       # 固定 alpha 以减少搜索空间
    }
    
    # 生成所有组合
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Total combinations to test: {len(combinations)}")
    
    results_log = []
    
    # 初始化数据处理器
    dp = DataProcessor(csv_path)
    
    # 为了速度，选择一些代表性赛季进行评估
    # 涵盖: Rank Rule (S1, S2, S30), Percentage Rule (S10, S20)
    # 既有早期也有近期，既有Rank也有Share
    # test_seasons = [1, 2, 10, 20, 30] 
    # 或者运行全量 (更准确)
    test_seasons = dp.get_seasons() 
    print(f"Testing on seasons: {test_seasons}")
    
    # 2. 遍历参数组合
    for i, params_dict in enumerate(combinations):
        print(f"\nTesting combination {i+1}/{len(combinations)}: {params_dict}")
        
        # 使用自定义参数创建模型
        # 注意: 搜索时适当减少粒子数以加速 (例如 200 或 500)
        # 正式运行时再用 1000
        current_params = ModelParams(
            n_particles=200, 
            rho=params_dict['rho'],
            gamma=params_dict['gamma'],
            alpha_discriminant=params_dict['alpha_discriminant']
        )
        
        model = SMCInverse(dp, current_params)
        
        try:
            # 运行模型
            results = model.run_all_seasons(seasons=test_seasons, verbose=False)
            
            # 评估结果
            analyzer = ResultAnalyzer(model, results)
            metrics = analyzer.compute_overall_hit_rate()
            consistency = analyzer.compute_consistency_metrics()
            
            # 记录得分
            score = {
                'rho': params_dict['rho'],
                'gamma': params_dict['gamma'],
                'alpha': params_dict['alpha_discriminant'],
                'top1_hit': metrics['overall_hit_rate_top1'],
                'top3_hit': metrics['overall_hit_rate_top3'],
                'posterior_consistency': consistency['overall_posterior_consistency'],
                'map_match_rate': consistency['overall_map_match_rate']
            }
            results_log.append(score)
            
            print(f"  -> Top-3 Hit: {score['top3_hit']:.2%}, Consistency: {score['posterior_consistency']:.2%}")
            
        except Exception as e:
            print(f"  -> Error: {e}")
            import traceback
            traceback.print_exc()

    # 5. 找出最佳结果
    if not results_log:
        print("No results collected.")
        return

    df = pd.DataFrame(results_log)
    
    # 定义综合得分 (Weighted Score)
    # 权重: Top-3 Hit (0.5) + MAP Match (0.3) + Posterior Consistency (0.2)
    # 理由: Top-3 最硬，MAP Match 次之，Posterior 是软指标
    df['weighted_score'] = (
        df['top3_hit'] * 0.8 + 
        df['map_match_rate'] * 0.1 + 
        df['posterior_consistency'] * 0.1
    )
    
    best_run = df.loc[df['weighted_score'].idxmax()]
    
    print("\n" + "="*50)
    print("Optimization Complete!")
    print(f"Time elapsed: {time.time() - start_time:.2f}s")
    print("Best Parameters found:")
    print(f"  rho (Memory): {best_run['rho']}")
    print(f"  gamma (Judge): {best_run['gamma']}")
    print(f"  alpha (Discriminant): {best_run['alpha']}")
    print("-" * 30)
    print(f"Best Top-3 Hit Rate: {best_run['top3_hit']:.2%}")
    print(f"Best MAP Match Rate: {best_run['map_match_rate']:.2%}")
    print(f"Best Consistency:    {best_run['posterior_consistency']:.2%}")
    print("="*50)
    
    # 保存结果到 CSV，方便做图
    df.to_csv('output/optimization_results.csv', index=False)
    print("Results saved to output/optimization_results.csv")
    
    return best_run

if __name__ == '__main__':
    optimize_hyperparameters()
