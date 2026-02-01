"""Generate complete controversy ranking including known cases."""
import pandas as pd
import json

# 读取发现的案例
df = pd.read_csv('output/discovered_controversial.csv')

# 读取已知案例的争议分数
with open('output/controversial_cases_details.json', 'r') as f:
    known = json.load(f)

# 添加4个已知案例
known_rows = []
for name, data in known.items():
    known_rows.append({
        'name': name,
        'season': data['season'],
        'final_place': data['computed_final_place'],
        'avg_judge_rank': data['avg_judge_rank'],
        'times_lowest_judge': data['times_lowest_judge'],
        'weeks_participated': data['weeks_participated'],
        'controversy_score': data['controversy_score'],
        'is_known_case': True
    })

df['is_known_case'] = False
df_known = pd.DataFrame(known_rows)
df_all = pd.concat([df, df_known], ignore_index=True)

# 按争议分数排序
df_all = df_all.sort_values('controversy_score', ascending=False).reset_index(drop=True)

# 显示TOP-15
print('='*80)
print('COMPLETE CONTROVERSY RANKING (Including Known Cases)')
print('='*80)
print()

for i, row in df_all.head(15).iterrows():
    marker = '*** KNOWN ***' if row['is_known_case'] else ''
    print(f"{i+1:2}. {row['name']:25} (S{row['season']:2}) | Final={row['final_place']:2} | AvgJudge={row['avg_judge_rank']:.1f} | Score={row['controversy_score']:.3f} {marker}")

# 保存完整排名
df_all.to_csv('output/full_controversy_ranking.csv', index=False)
print()
print('Saved to: output/full_controversy_ranking.csv')

# 检查已知案例的排名
print()
print('='*80)
print('KNOWN CASES RANKING POSITIONS')
print('='*80)
for name in ['Bobby Bones', 'Jerry Rice', 'Bristol Palin', 'Billy Ray Cyrus']:
    matches = df_all[df_all['name'] == name]
    if not matches.empty:
        pos = matches.index[0] + 1
        row = matches.iloc[0]
        in_top10 = "YES ✓" if pos <= 10 else "NO"
        print(f"{name}: Rank #{pos} | Score={row['controversy_score']:.3f} | Top-10: {in_top10}")
