
import pandas as pd
import numpy as np

def verify_sources():
    print("Verifying Data Sources...")
    
    # Load weekly results
    df = pd.read_csv('output/q2_weekly_results.csv')
    
    # 1. Overall Stats
    avg_agreement = df['agreement_rate'].mean()
    print(f"Overall Agreement Rate: {avg_agreement:.1%}")
    
    avg_fan_rank = df['fan_align_rank'].mean()
    avg_fan_pct = df['fan_align_pct'].mean()
    print(f"Fan Alignment: Rank={avg_fan_rank:.3f}, Pct={avg_fan_pct:.3f}, Delta={avg_fan_pct - avg_fan_rank:.3f}")
    
    avg_judge_rank = df['judge_align_rank'].mean()
    avg_judge_pct = df['judge_align_pct'].mean()
    print(f"Judge Alignment: Rank={avg_judge_rank:.3f}, Pct={avg_judge_pct:.3f}, Delta={avg_judge_rank - avg_judge_pct:.3f}")
    
    # 2. Three Eras Analysis
    # Era 1: S1-S2
    era1 = df[df['season'].isin([1, 2])]
    print(f"\nEra 1 (S1-S2) Agreement: {era1['agreement_rate'].mean():.1%}")
    
    # Era 2: S3-S27
    era2 = df[(df['season'] >= 3) & (df['season'] <= 27)]
    print(f"Era 2 (S3-S27) Agreement: {era2['agreement_rate'].mean():.1%}")
    
    # Era 3: S28+
    era3 = df[df['season'] >= 28]
    print(f"Era 3 (S28+) Agreement: {era3['agreement_rate'].mean():.1%}")

if __name__ == "__main__":
    verify_sources()
