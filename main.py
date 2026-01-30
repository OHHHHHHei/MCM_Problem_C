"""
SMC-Inverse 主程序入口
序贯蒙特卡洛粒子滤波系统 - 《与星共舞》隐状态反演

Usage:
    python main.py                    # 运行所有赛季 (默认500粒子)
    python main.py --seasons 27 28    # 运行指定赛季
    python main.py --particles 1000   # 使用1000个粒子
    python main.py --test             # 测试模式 (少量粒子，快速验证)
"""

import argparse
import os
import sys
import json
import time
from typing import List, Optional

from data_processor import DataProcessor
from smc_inverse import SMCInverse, ModelParams, create_model
from analysis import ResultAnalyzer, generate_text_report


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='SMC-Inverse: Sequential Monte Carlo for DWTS Latent State Inversion'
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default='2026_MCM_Problem_C_Data.csv',
        help='Path to CSV data file'
    )
    
    parser.add_argument(
        '--seasons', '-s',
        type=int,
        nargs='*',
        default=None,
        help='Specific seasons to run (default: all)'
    )
    
    parser.add_argument(
        '--particles', '-p',
        type=int,
        default=500,
        help='Number of particles (default: 500)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode with fewer particles'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode (less verbose output)'
    )
    
    # 模型参数
    parser.add_argument('--kappa', type=float, default=0.3, help='Performance shock coefficient')
    parser.add_argument('--delta', type=float, default=1.5, help='Shock threshold')
    parser.add_argument('--rho', type=float, default=0.6, help='Memory decay coefficient')
    parser.add_argument('--gamma', type=float, default=0.5, help='Judge influence coefficient')
    
    return parser.parse_args()


def run_model(args) -> dict:
    """运行模型的主函数"""
    
    # 测试模式
    if args.test:
        args.particles = 100
        args.seasons = [27]  # 只运行第27季
        print("[TEST MODE] Running with 100 particles on Season 27 only")
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查数据文件
    if not os.path.exists(args.data):
        print(f"Error: Data file not found: {args.data}")
        sys.exit(1)
    
    print("=" * 60)
    print("SMC-Inverse: Latent State Inversion System")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data file: {args.data}")
    print(f"  Particles: {args.particles}")
    print(f"  Output dir: {args.output}")
    
    # 加载数据
    print("\nLoading data...")
    dp = DataProcessor(args.data)
    available_seasons = dp.get_seasons()
    print(f"  Available seasons: {available_seasons[0]} - {available_seasons[-1]}")
    print(f"  Total contestants: {len(dp.contestants)}")
    
    # 确定要运行的赛季
    if args.seasons:
        seasons = [s for s in args.seasons if s in available_seasons]
        if not seasons:
            print(f"Error: None of the specified seasons are available")
            sys.exit(1)
    else:
        seasons = available_seasons
    
    print(f"  Seasons to process: {seasons}")
    
    # 创建模型
    print("\nInitializing model...")
    params = ModelParams(
        n_particles=args.particles,
        kappa=args.kappa,
        delta=args.delta,
        rho=args.rho,
        gamma=args.gamma
    )
    model = SMCInverse(dp, params)
    
    # 运行
    print("\n" + "-" * 60)
    print("Running particle filter...")
    print("-" * 60)
    
    start_time = time.time()
    
    results = model.run_all_seasons(seasons, verbose=not args.quiet)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    
    # 分析
    print("\n" + "-" * 60)
    print("Analyzing results...")
    print("-" * 60)
    
    analyzer = ResultAnalyzer(model, results)
    
    # 生成报告
    report = generate_text_report(analyzer)
    print(report)
    
    # 保存结果
    print("\n" + "-" * 60)
    print("Exporting results...")
    print("-" * 60)
    
    # 保存文本报告
    report_path = os.path.join(args.output, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  Text report: {report_path}")
    
    # 保存JSON结果
    analyzer.export_results(args.output)
    
    # 保存详细投票份额估计
    estimates_path = os.path.join(args.output, 'vote_share_estimates.json')
    estimates_export = {}
    for season, result in results.items():
        estimates_export[str(season)] = {}
        for week, est in result.get('weekly_estimates', {}).items():
            estimates_export[str(season)][str(week)] = {
                name: {'mean': float(v['mean']), 'std': float(v['std'])}
                for name, v in est.items()
            }
    
    with open(estimates_path, 'w', encoding='utf-8') as f:
        json.dump(estimates_export, f, indent=2, ensure_ascii=False)
    print(f"  Vote estimates: {estimates_path}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    
    return results


def main():
    """主入口"""
    args = parse_args()
    
    try:
        results = run_model(args)
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
