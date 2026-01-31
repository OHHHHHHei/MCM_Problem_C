import pandas as pd
import os

target_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_overnight_results.csv'))
output_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_highlights.csv'))

def extract_highlights():
    if not os.path.exists(target_file):
        print(f"Error: {target_file} not found.")
        return

    print(f"Loading {target_file}...")
    df = pd.read_csv(target_file)
    
    # 1. 预处理
    # 添加原始行号 (Line Number = Index + 2, because of 0-based index and 1 header row)
    df['original_line'] = df.index + 2
    
    # 确保有 weighted_score
    if 'weighted_score' not in df.columns:
        if 'top3_hit' in df.columns:
            df['weighted_score'] = df['top3_hit'] 
            if 'map_match_rate' in df.columns and 'posterior_consistency' in df.columns:
                 df['weighted_score'] = df['top3_hit'] * 0.5 + df['map_match_rate'] * 0.3 + df['posterior_consistency'] * 0.2

    # 2. 提取各类 Top 榜单
    
    # Category A: Top-1 Hit Rate (精准预测冠军/倒数第一)
    top1_best = df.sort_values(by='top1_hit', ascending=False).head(5)
    top1_best['Category'] = 'Top-1 Best (Precision)'
    
    # Category B: Top-3 Hit Rate (危险区预测专家)
    top3_best = df.sort_values(by='top3_hit', ascending=False).head(5)
    top3_best['Category'] = 'Top-3 Best (Recall)'
    
    # Category C: Weighted Score (综合表现最强)
    weighted_best = df.sort_values(by='weighted_score', ascending=False).head(10)
    weighted_best['Category'] = 'Overall Best (Weighted)'
    
    # Category D: Special - Highest MAP Match Rate (机制最还原)
    # MAP = Maximum A Posteriori，代表模型仅凭"所有粒子的平均值"就能算出正确结果的能力
    # 这代表了模型 captured the "Deterministic Signal" best
    map_best = df.sort_values(by='map_match_rate', ascending=False).head(3)
    map_best['Category'] = 'Mechanistic Best (MAP Match)'
    
    # Category E: Special - High Gamma Contrast (高从众心理对照组)
    # 找出 gamma >= 0.7 的最佳结果，用于论文对比 "粉丝独立 vs 盲从"
    high_gamma_df = df[df['gamma'] >= 0.7]
    if not high_gamma_df.empty:
        contrast_best = high_gamma_df.sort_values(by='weighted_score', ascending=False).head(3)
        contrast_best['Category'] = 'Contrast: High Judge Influence'
    else:
        contrast_best = pd.DataFrame()

    # 3. 合并
    frames = [weighted_best, top3_best, top1_best, map_best, contrast_best]
    result = pd.concat(frames)
    
    # 去重 (保留第一次出现的，因为顺序是按重要性加进去的)
    result = result.drop_duplicates(subset=['rho', 'gamma', 'delta', 'alpha', 'kappa', 'beta_judge'])
    
    # 重新排序列，让 Category 在最前
    cols = ['Category', 'original_line'] + [c for c in result.columns if c not in ['Category', 'original_line']]
    result = result[cols]
    
    # 保存
    result.to_csv(output_file, index=False)
    print(f"\nSuccess! Highlights extracted to {output_file}")
    print(f"Total unique rows: {len(result)}")
    print("-" * 30)
    print(result[['Category', 'original_line', 'top3_hit', 'weighted_score']].to_string())

if __name__ == '__main__':
    extract_highlights()
