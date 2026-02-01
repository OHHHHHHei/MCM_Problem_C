
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrix

def calc_vif():
    df = pd.read_csv('output/q3_panel_data_7dim.csv')
    
    # We are checking multicollinearity for the predictors in the most complex model (Track B2)
    # Predictors: judge_score_z + age_std + C(industry) + C(week_cat)
    # week_cat usually controls for time, but might be collinear with age if everyone gets older? 
    # Actually age is static per season usually, or age during season.
    # Let's focus on the main variables of interest first: judge_score_z, age_std, industry.
    
    # Using patsy to create the design matrix
    # We drop one industry as reference (automatically done by dmatrix with Intercept)
    formula = "judge_score_z + age_std + C(industry_7dim, Treatment(reference='Other'))"
    
    # dmatrix will create the design matrix X
    # return_type='dataframe' to keep column names
    X = dmatrix(formula, df, return_type='dataframe')
    
    print(f"Design Matrix Shape: {X.shape}")
    
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Filter to show relevant variables (Intercept usually high but irrelevant)
    # print all for now
    print("\nCalculated VIF Results:")
    print(vif_data.round(2))
    
    # Checks
    max_vif = vif_data[vif_data["Variable"] != "Intercept"]["VIF"].max()
    print(f"\nMax VIF (excluding Intercept): {max_vif:.2f}")

if __name__ == "__main__":
    calc_vif()
