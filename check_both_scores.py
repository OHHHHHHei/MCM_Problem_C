"""查找同时具有 S1 和 S2 指标的选手"""
import pandas as pd

df = pd.read_csv('output/controversy_v3_full.csv')

# 找出 S1 > 0.05 且 S2 > 0.5 的选手
both = df[(df['S1_populist'] > 0.05) & (df['S2_robbed'] > 0.5)]
both = both.sort_values('S1_populist', ascending=False)

print('='*70)
print('CONTESTANTS WITH BOTH S1 > 0.05 AND S2 > 0.5')
print('='*70)
print(f'Total: {len(both)} contestants')
print()

for _, row in both.head(15).iterrows():
    print(f"{row['name']:25} (S{row['season']:2}) | P={row['final_place']:2} | "
          f"S1={row['S1_populist']:.3f} | S2={row['S2_robbed']:.3f} | "
          f"R_avg={row['R_avg']:.1f} | Z_last={row['Z_last']:.2f}")
