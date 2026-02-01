"""
Controversy Quadrant Plot V2

改进版：
1. 移除彩色区域，只保留颜色渐变
2. 只标注重要的点（Top-5 每维度 + 官方案例）
3. 使用 adjustText 自动避免标签重叠
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# 尝试导入 adjustText，如果没有则用简单方案
try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Note: adjustText not installed, using manual label placement")

plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def plot_controversy_quadrant_v2(csv_path='output/controversy_v3_full.csv', 
                                  output_path='output/controversy_quadrant.png'):
    """生成争议象限图 V2"""
    
    df = pd.read_csv(csv_path)
    
    # 计算归一化争议指数
    S1_max = df['S1_populist'].max()
    S2_max = df['S2_robbed'].max()
    
    df['S1_norm'] = df['S1_populist'] / S1_max
    df['S2_norm'] = df['S2_robbed'] / S2_max
    df['controversy_index'] = np.sqrt(df['S1_norm']**2 + df['S2_norm']**2)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 主散点图 - 使用颜色渐变表示争议程度
    scatter = ax.scatter(
        df['S1_populist'], 
        df['S2_robbed'],
        c=df['controversy_index'],
        cmap='YlOrRd',
        s=50,
        alpha=0.6,
        edgecolors='white',
        linewidth=0.3
    )
    
    # 颜色条
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Controversy Index\n(Normalized Distance from Origin)', fontsize=11)
    
    # 定义需要标注的点
    known_cases = {
        'Bobby Bones': {'season': 27, 'color': '#c62828', 'priority': 1},
        'Jerry Rice': {'season': 2, 'color': '#c62828', 'priority': 1},
        'Bristol Palin': {'season': 11, 'color': '#c62828', 'priority': 1},
        'Billy Ray Cyrus': {'season': 4, 'color': '#e65100', 'priority': 2},
    }
    
    # S1 Top-3
    s1_top3 = df.nlargest(3, 'S1_populist')['name'].tolist()
    # S2 Top-3
    s2_top3 = df.nlargest(3, 'S2_robbed')['name'].tolist()
    
    # 收集要标注的点
    texts = []
    
    # 1. 标注官方案例 (大星星)
    for name, info in known_cases.items():
        row = df[(df['name'] == name) & (df['season'] == info['season'])]
        if not row.empty:
            x, y = row['S1_populist'].values[0], row['S2_robbed'].values[0]
            ax.scatter(x, y, s=300, marker='*', color=info['color'], 
                      edgecolors='black', linewidth=1.5, zorder=10)
    
    # 2. 标注 S1 Top-3 (非官方案例)
    for name in s1_top3:
        if name not in known_cases:
            row = df[df['name'] == name].iloc[0]
            x, y = row['S1_populist'], row['S2_robbed']
            ax.scatter(x, y, s=120, marker='o', facecolors='none', 
                      edgecolors='#d32f2f', linewidth=2, zorder=8)
    
    # 3. 标注 S2 Top-3
    for name in s2_top3:
        if name not in known_cases and name not in s1_top3:
            row = df[df['name'] == name].iloc[0]
            x, y = row['S1_populist'], row['S2_robbed']
            ax.scatter(x, y, s=120, marker='o', facecolors='none', 
                      edgecolors='#1565c0', linewidth=2, zorder=8)
    
    # 轴标签 (只保留符号)
    ax.set_xlabel(r'$S_1$ (Populist Score)', fontsize=12)
    ax.set_ylabel(r'$S_2$ (Robbed Score)', fontsize=12)
    ax.set_title('Controversy Quadrant', fontsize=14, fontweight='bold')
    
    # 网格
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(-0.02, S1_max * 1.15)
    ax.set_ylim(-0.15, S2_max * 1.1)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#c62828', 
               markersize=15, label='Known Cases (Problem Statement)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#d32f2f', markeredgewidth=2, markersize=10, 
               label='Top-3 S1 (Populist)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#1565c0', markeredgewidth=2, markersize=10, 
               label='Top-3 S2 (Robbed)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    

    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # 保存带争议指数的数据
    df_export = df[['name', 'season', 'final_place', 'S1_populist', 'S2_robbed', 
                    'S1_norm', 'S2_norm', 'controversy_index', 'is_known_case']]
    df_export = df_export.sort_values('controversy_index', ascending=False)
    df_export.to_csv('output/controversy_with_index.csv', index=False)
    print("Saved: output/controversy_with_index.csv")


if __name__ == '__main__':
    plot_controversy_quadrant_v2()
