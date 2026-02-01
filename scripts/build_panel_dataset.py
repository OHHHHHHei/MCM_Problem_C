
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from scipy.stats import zscore

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_processor import DataProcessor
from core.smc_inverse import SMCInverse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_FILE = 'output/q3_panel_data_7dim.csv'
CSV_PATH = 'data/raw/2026_MCM_Problem_C_Data.csv'

def safe_logit(p, epsilon=1e-6):
    """
    Apply logit transform with clamping to avoid infinity.
    logit(p) = log(p / (1-p))
    """
    p_clamped = np.clip(p, epsilon, 1 - epsilon)
    return np.log(p_clamped / (1 - p_clamped))

def map_industry_q1(raw_ind):
    """
    Map raw industry string to Q1's 7 dimensions.
    Logic from core/data_processor.py
    """
    if pd.isna(raw_ind): return "Other"
    raw = str(raw_ind).lower()
    
    # Priority order matches Q1 logic
    categories = ['Actor/Actress', 'Athlete', 'Singer/Rapper', 'TV Personality', 'Model', 'Comedian']
    
    for cat in categories:
        if cat.lower() in raw:
            return cat
            
    return "Other"

def build_panel_data():
    logger.info("Starting Q3 Panel Data Construction...")
    
    # 1. Initialize Processor
    dp = DataProcessor(CSV_PATH)
    smc = SMCInverse(dp)
    
    seasons = dp.get_seasons()
    all_rows = []
    
    # Load raw dataframe for static features lookup
    raw_df = pd.read_csv(CSV_PATH)
    # create lookup: name -> {partner, age, industry}
    # Columns: celebrity_name, ballroom_partner, celebrity_industry, celebrity_age_during_season
    static_lookup = {}
    for _, row in raw_df.iterrows():
        name = row['celebrity_name']
        static_lookup[name] = {
            'partner': row['ballroom_partner'],
            'age': row['celebrity_age_during_season'],
            'industry': row['celebrity_industry'],
            'industry_7dim': map_industry_q1(row['celebrity_industry'])
        }
    
    # 2. Iterate through all seasons
    for season in tqdm(seasons, desc="Processing Seasons"):
        try:
            res = smc.run_season(season, verbose=False)
        except Exception as e:
            logger.error(f"Failed to run SMC for Season {season}: {e}")
            continue
            
        if not res or 'weekly_estimates' not in res:
            logger.warning(f"No results for Season {season}")
            continue
            
        weekly_data = res['weekly_estimates'] # {week: {name: {'mean': ..., 'std': ...}}}
        
        # Get active contestant name mapping if needed
        contestants = dp.get_contestants_in_season(season)
        c_map = {c.name: c for c in contestants}

        # 3. Process each week
        sorted_weeks = sorted(weekly_data.keys())
        
        for week in sorted_weeks:
            week_estimates = weekly_data[week] # {name: stats}
            active_names = list(week_estimates.keys())
            
            # Retrieve raw judge scores
            raw_scores = {}
            for name in active_names:
                c_obj = c_map.get(name)
                if not c_obj: continue
                score = dp.get_weekly_total_score(c_obj, week)
                if score is not None:
                    raw_scores[name] = score
                else:
                    raw_scores[name] = np.nan
            
            valid_scores = [s for s in raw_scores.values() if not np.isnan(s)]
            if not valid_scores:
                continue
                
            mean_score = np.mean(valid_scores)
            std_score = np.std(valid_scores)
            if std_score < 1e-9: std_score = 1.0
            
            # 4. Build Rows
            for name, stats in week_estimates.items():
                if name not in static_lookup:
                    continue
                
                meta = static_lookup[name]
                raw_score = raw_scores.get(name, np.nan)
                
                if np.isnan(raw_score):
                    continue
                    
                judge_z = (raw_score - mean_score) / std_score
                pi_mean = stats['mean']
                
                row = {
                    'season': season,
                    'week': week,
                    'contestant': name,
                    'partner': meta.get('partner', 'Unknown'),
                    'age': meta.get('age', 30),
                    'industry': meta.get('industry', 'Other'),
                    'industry_7dim': meta.get('industry_7dim', 'Other'),
                    
                    'judge_score_raw': raw_score,
                    'judge_score_z': judge_z,
                    
                    'vote_share_pi': pi_mean,
                    'vote_share_logit': safe_logit(pi_mean)
                }
                all_rows.append(row)

    # 5. Save dataframe
    df = pd.DataFrame(all_rows)
    
    # Clean Data
    df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(30)
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    df['vote_share_logit'] = np.clip(df['vote_share_logit'], -10, 10)
    
    df['season_id'] = df['season'].astype(str)
    df['star_id'] = df['contestant']
    df['pro_id'] = df['partner']
    
    logger.info(f"Construction Complete. Rows: {len(df)}")
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Panel data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_panel_data()
