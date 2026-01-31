import pandas as pd
import os

target_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_overnight_results.csv'))
fallback_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output/optimization_aggressive_results.csv'))

if os.path.exists(target_file):
    csv_path = target_file
    print(f"Found Overnight Results: {csv_path}")
elif os.path.exists(fallback_file):
    csv_path = fallback_file
    print(f"Found Aggressive Results (Fallback): {csv_path}")
else:
    print("No optimization result files found in output/ directory.")
    print("Files found:", os.listdir('output'))
    exit(1)

try:
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows.")
    
    # Create weighted score if missing
    if 'weighted_score' not in df.columns:
        if 'top3_hit' in df.columns:
            df['weighted_score'] = df['top3_hit'] 
            if 'map_match_rate' in df.columns and 'posterior_consistency' in df.columns:
                 df['weighted_score'] = df['top3_hit'] * 0.5 + df['map_match_rate'] * 0.3 + df['posterior_consistency'] * 0.2
    
    # Add Traceability: Original Line Number (1-based, accounts for header)
    df['csv_line_number'] = df.index + 2

    # Sort
    df_sorted = df.sort_values(by='weighted_score', ascending=False)
    
    print("\n" + "="*50)
    print("TOP 5 PARAMETER COMBINATIONS (With CSV Line Numbers)")
    print("="*50)
    # Reorder columns to show line number first
    cols = ['csv_line_number'] + [c for c in df_sorted.columns if c != 'csv_line_number']
    print(df_sorted[cols].head(5).to_string(index=False))
    
    # Best Weighted (Already found)
    best_weighted = df_sorted.iloc[0]
    
    # Best Top-1
    df_top1 = df.sort_values(by='top1_hit', ascending=False)
    best_top1 = df_top1.iloc[0]
    
    # Best Top-3
    df_top3 = df.sort_values(by='top3_hit', ascending=False)
    best_top3 = df_top3.iloc[0]

    print("\n" + "="*50)
    print(f"HIGHEST TOP-1 HIT RATE (Line {best_top1['csv_line_number']})")
    print("="*50)
    for k, v in best_top1.items():
        print(f"{k}: {v}")

    print("\n" + "="*50)
    print(f"HIGHEST TOP-3 HIT RATE (Line {best_top3['csv_line_number']})")
    print("="*50)
    for k, v in best_top3.items():
        print(f"{k}: {v}")
        
    print("\n" + "="*50)
    print(f"BEST WEIGHTED SCORE (Line {best_weighted['csv_line_number']})")
    print("="*50)
    for k, v in best.items():
        print(f"{k}: {v}")

except Exception as e:
    print(f"Error analyzing CSV: {e}")
