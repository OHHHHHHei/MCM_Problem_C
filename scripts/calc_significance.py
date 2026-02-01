
import pandas as pd
import numpy as np
from scipy import stats

def calc_sig():
    df = pd.read_csv('output/q3_coefficients.csv')
    
    # Filter for Judge and Fan_Bias models
    df = df[df['Model'].isin(['Judge', 'Fan_Bias'])]
    
    table_rows = []
    
    industries = [
        'Politician', 'Journalist', 'Reality Star', 'Athlete', 
        'Musician', 'TV Personality', 'Model', 'Comedian'
    ]
    
    print("| 行业 | Judge beta | Fan beta | Sig Judge | Sig Fan |")
    print("|---|---|---|---|---|")
    
    stats_data = []

    for ind in industries:
        term = f"C(industry)[T.{ind}]"
        
        # Get Judge
        j_row = df[(df['Model']=='Judge') & (df['Term'] == term)]
        if len(j_row) == 0: continue
        j_coef = j_row.iloc[0]['Coef']
        j_se = j_row.iloc[0]['SE']
        j_z = j_coef / j_se
        j_p = 2 * (1 - stats.norm.cdf(abs(j_z)))
        
        # Get Fan
        f_row = df[(df['Model']=='Fan_Bias') & (df['Term'] == term)]
        if len(f_row) == 0: continue
        f_coef = f_row.iloc[0]['Coef']
        f_se = f_row.iloc[0]['SE']
        f_z = f_coef / f_se
        f_p = 2 * (1 - stats.norm.cdf(abs(f_z)))
        
        delta = f_coef - j_coef
        
        def star(p):
            if p < 0.001: return "***"
            if p < 0.01: return "**"
            if p < 0.05: return "*"
            if p < 0.1: return "."
            return "ns"
            
        print(f"| {ind} | {j_coef:.2f} ({star(j_p)}) | {f_coef:.2f} ({star(f_p)}) | {j_p:.4f} | {f_p:.4f} |")
        
        stats_data.append({
            'Industry': ind,
            'Judge': f"{j_coef:.2f}{star(j_p)}",
            'Fan': f"{f_coef:.2f}{star(f_p)}",
            'Delta': f"{delta:.2f}"
        })

if __name__ == "__main__":
    calc_sig()
