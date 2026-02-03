import os
import sys
import pandas as pd
from collections import defaultdict

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor

def verify_stats():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    dp = DataProcessor(csv_path)
    
    # 1. Seasons
    # dp.df contains the raw processing dataframe
    seasons = sorted(dp.df['season'].unique())
    n_seasons = len(seasons)
    
    # 2. Unique Contestants
    # Use standard column names from raw CSV via dp.df
    # Clean column names first just in case
    dp.df.columns = [c.strip().replace('ï»¿', '') for c in dp.df.columns]
    
    unique_contestants = dp.df[['season', 'celebrity_name']].drop_duplicates()
    n_contestants = len(unique_contestants)
    
    # 3. Elimination Events
    # Count rows where 'result' indicates elimination
    # DataProcessor uses `get_elimination_events(season)` which parses weekly data.
    # Let's count them properly via the processor logic.
    total_elim_events = 0
    total_eliminated_people = 0
    
    contestants_per_season = []
    
    for s in seasons:
        # Contestants per season
        c_in_season = unique_contestants[unique_contestants['season'] == s]
        contestants_per_season.append(len(c_in_season))
        
        # Elim events
        events = dp.get_elimination_events(s)
        total_elim_events += len(events)
        for e in events:
            total_eliminated_people += len(e.eliminated_set)
            
    avg_contestants = sum(contestants_per_season) / n_seasons
    
    print("\n=== Dataset Verification ===")
    print(f"Total Seasons: {n_seasons}")
    print(f"Total Unique Contestants: {n_contestants}")
    print(f"Total Elimination Events (Process Rounds): {total_elim_events}")
    print(f"Total Eliminated People: {total_eliminated_people}")
    print(f"Average Contestants per Season: {avg_contestants:.2f}")
    
    # Verify claims
    print("\n=== Claims Verification ===")
    print(f"Claim: 34 Seasons -> {'CORRECT' if n_seasons == 34 else f'MISMATCH ({n_seasons})'}")
    print(f"Claim: 421 Contestants -> {'CORRECT' if n_contestants == 421 else f'MISMATCH ({n_contestants})'}")
    print(f"Claim: 298 Elimination Events -> Check: Is it Process Rounds ({total_elim_events}) or People ({total_eliminated_people})?")
    print(f"Claim: 12.4 Avg/Season -> {'CORRECT' if abs(avg_contestants - 12.4) < 0.1 else f'MISMATCH ({avg_contestants:.2f})'}")

if __name__ == "__main__":
    verify_stats()
