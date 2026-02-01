
import pandas as pd
import numpy as np

def validate_data():
    df = pd.read_csv('output/q3_panel_data.csv')
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # 1. Partner Analysis
    unique_partners = df['partner'].unique()
    print(f"\nUnique Partners ({len(unique_partners)}): {unique_partners[:10]}...")
    
    # Check if partner name looks valid (not 'Unknown' or nan)
    unknown_partners = df[df['partner'] == 'Unknown']
    if len(unknown_partners) > 0:
        print(f"WARNING: {len(unknown_partners)} rows have 'Unknown' partner.")
        
    # Check Partner Variance impact potential
    # If every star has a unique partner (1-to-1 mapping), then Star and Pro are confounded
    # and we can't separate them (ICC would be huge or undefined depending on structure).
    # But usually Pros repeat across seasons.
    pro_counts = df.groupby('partner')['season_id'].nunique()
    print(f"\nPros appearing in multiple seasons: {len(pro_counts[pro_counts > 1])} / {len(pro_counts)}")
    
    # 2. Judge Score Analysis
    print(f"\nJudge Score Z Analysis:")
    print(df['judge_score_z'].describe())
    
    # 3. Fan Vote Analysis
    print(f"\nFan Vote Logit Analysis:")
    print(df['vote_share_logit'].describe())
    
    # Check correlations
    corr = df[['judge_score_z', 'vote_share_logit', 'age_std']].corr()
    print("\nCorrelation Matrix:")
    print(corr)

if __name__ == "__main__":
    validate_data()
