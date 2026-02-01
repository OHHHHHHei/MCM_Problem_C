
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import os

OUTPUT_DIR = 'output'
DATA_FILE = 'output/q3_panel_data.csv'

def fit_lmm(df, formula, groups, re_formula=None):
    """
    Wrapper for MixedLM to handle convergence issues gracefully.
    """
    print(f"Fitting: {formula}")
    model = smf.mixedlm(formula, df, groups=groups, re_formula=re_formula)
    try:
        # Try default fit
        result = model.fit(method=["lbfgs", "cg"], maxiter=500) # Increased maxiter
    except Exception as e:
        print(f"  Fit failed: {e}. Retrying with simpler optimizer...")
        try:
             result = model.fit(reml=False) # Try ML instead of REML
        except:
             return None
             
    # Singularity warning is common in LMM, just proceed if we have params
    return result

def safe_get_scalar(val):
    """Ensure value is a scalar float."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)

def decompose_variance(model_result):
    """
    Extract Variance Components and calculate ICC.
    """
    if model_result is None: 
        return {}
        
    try:
        cov_params = model_result.cov_params()
        scale = model_result.scale # Residual Variance
        
        re_vars = {}
        # Iterate and extract diagonal terms (variances)
        # Statsmodels naming can be 'Group Var' or specific names
        for k in cov_params.index:
            if 'Group Var' in k or 'Var' in k:
                val = cov_params.loc[k]
                # In case cov_params returns DataFrame/Series structure
                if isinstance(val, pd.Series):
                    val = val.iloc[0] # Take the value
                re_vars['Group'] = val # For single group model, this is the main RE
        
        # Calculate total
        group_var = re_vars.get('Group', 0)
        total_var = scale + group_var
        
        iccs = {
            'Group': safe_get_scalar(group_var / total_var) if total_var > 0 else 0,
            'Residual': safe_get_scalar(scale / total_var) if total_var > 0 else 0
        }
        return iccs
    except Exception as e:
        print(f"  Variance decomposition error: {e}")
        return {'Group': 0.0, 'Residual': 1.0}

def run_analysis():
    print("Loading Panel Data...")
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    df = pd.read_csv(DATA_FILE)
    
    # Pre-processing
    df['season_id'] = df['season_id'].astype(str)
    df['industry'] = df['industry'].astype('category')
    df['week_cat'] = df['week'].astype(str)
    
    # Standardize Age
    df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
    
    # Standardize Y^F for comparison
    df['vote_share_logit_z'] = (df['vote_share_logit'] - df['vote_share_logit'].mean()) / df['vote_share_logit'].std()
    
    results_summary = []
    
    # =========================================================================
    # Track A
    # =========================================================================
    print("\n--- Track A: Meritocracy (Judge Score) ---")
    f_A = "judge_score_z ~ age_std + C(industry) + C(week_cat)"
    res_A = fit_lmm(df, f_A, groups=df["pro_id"])
    
    icc_A = {}
    if res_A:
        # print(res_A.summary()) 
        icc_A = decompose_variance(res_A)
        print(f"Track A Pro ICC: {icc_A.get('Group', 0):.4f}")
    
    # =========================================================================
    # Track B1
    # =========================================================================
    print("\n--- Track B1: Fan Popularity (Total Vote Share) ---")
    f_B1 = "vote_share_logit_z ~ age_std + C(industry) + C(week_cat)"
    res_B1 = fit_lmm(df, f_B1, groups=df["pro_id"])
    
    icc_B1 = {}
    if res_B1:
        icc_B1 = decompose_variance(res_B1)
        print(f"Track B1 Pro ICC: {icc_B1.get('Group', 0):.4f}")

    # =========================================================================
    # Track B2
    # =========================================================================
    print("\n--- Track B2: Fan Bias (Net Preference) ---")
    f_B2 = "vote_share_logit_z ~ judge_score_z + age_std + C(industry) + C(week_cat)"
    res_B2 = fit_lmm(df, f_B2, groups=df["pro_id"])
    
    icc_B2 = {}
    if res_B2:
        # print(res_B2.summary())
        icc_B2 = decompose_variance(res_B2)
        print(f"Track B2 Pro ICC: {icc_B2.get('Group', 0):.4f}")

    # =========================================================================
    # Export Results
    # =========================================================================
    rows = []
    models = {'Judge': res_A, 'Fan_Bias': res_B2}
    
    for m_name, res in models.items():
        if res is None: continue
        p = res.params
        e = res.bse
        ci = res.conf_int()
        
        for term in p.index:
            if 'age_std' in term or 'industry' in term:
                rows.append({
                    'Model': m_name,
                    'Term': term,
                    'Coef': p[term],
                    'SE': e[term],
                    'Lower': ci.loc[term, 0],
                    'Upper': ci.loc[term, 1]
                })
    
    res_df = pd.DataFrame(rows)
    res_df.to_csv('output/q3_coefficients.csv', index=False)
    
    # Save Variance Comparison
    var_rows = []
    if res_A: var_rows.append({'Model': 'Judge', 'Pro_ICC': icc_A.get('Group', 0), 'Residual': icc_A.get('Residual', 0)})
    if res_B1: var_rows.append({'Model': 'Fan_Total', 'Pro_ICC': icc_B1.get('Group', 0), 'Residual': icc_B1.get('Residual', 0)})
    if res_B2: var_rows.append({'Model': 'Fan_Bias', 'Pro_ICC': icc_B2.get('Group', 0), 'Residual': icc_B2.get('Residual', 0)})
    
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv('output/q3_variance_decomposition.csv', index=False)
    
    print("\nAnalysis Complete. Results saved.")

if __name__ == "__main__":
    run_analysis()
