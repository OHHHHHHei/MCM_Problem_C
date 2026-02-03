import pandas as pd
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_age():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'raw', '2026_MCM_Problem_C_Data.csv')
    df = pd.read_csv(csv_path, encoding='ISO-8859-1') # Handle encoding if needed
    
    # Filter unique contestants (names might repeat across weeks, usually we want unique people)
    # Assuming 'contestant_name' or similar column. Let's inspect columns first or just assume standard.
    # The file likely has 'Season', 'Name', 'Age', etc.
    # Use distinct Season + Name pairs
    
    # Rename columns to standard
    df.columns = [c.strip().replace('ï»¿', '') for c in df.columns]
    
    # Drop duplicates to get unique contestants per season
    # Using 'celebrity_name' and 'season' 
    if 'celebrity_age_during_season' not in df.columns:
        print("Columns after cleaning:", df.columns)
        return

    unique_contestants = df[['season', 'celebrity_name', 'celebrity_age_during_season']].drop_duplicates()
    
    mean_age = unique_contestants['celebrity_age_during_season'].mean()
    std_age = unique_contestants['celebrity_age_during_season'].std()
    median_age = unique_contestants['celebrity_age_during_season'].median()
    
    print(f"Unique Contestants Count: {len(unique_contestants)}")
    print(f"Mean Age: {mean_age:.4f}")
    print(f"Std Dev Age: {std_age:.4f}")
    print(f"Median Age: {median_age:.4f}")

if __name__ == "__main__":
    check_age()
