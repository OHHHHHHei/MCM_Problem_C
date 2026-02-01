
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os

OUTPUT_DIR = 'output'
PLOT_DIR = 'output/plots'
COEFF_FILE = 'output/q3_coefficients.csv'
VAR_FILE = 'output/q3_variance_decomposition.csv'

def plot_forest_comparison():
    if not os.path.exists(COEFF_FILE):
        return
        
    df = pd.read_csv(COEFF_FILE)
    
    # Process Term names for display
    # Remove C(...) and Reference level stuff
    df['Term_Clean'] = df['Term'].apply(
        lambda x: x.replace('C(industry)[T.', '').replace(']', '')
                   .replace('age_std', 'Age (Std)')
                   .replace('judge_score_z', 'Judge Score (Z)')
    )
    
    # Filter out irrelevant terms (like Intercept or Week if too many)
    df = df[~df['Term'].str.contains('week')]
    df = df[~df['Term'].str.contains('Intercept')]
    
    # Sort
    df = df.sort_values('Term_Clean')
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Create point plot
    # Dodge points to separate models
    models = df['Model'].unique()
    colors = {'Judge': '#1f77b4', 'Fan_Bias': '#d62728'}
    
    y_pos = np.arange(len(df['Term_Clean'].unique()))
    terms = df['Term_Clean'].unique()
    
    ax = plt.gca()
    
    offset = 0.2
    for i, model in enumerate(models):
        subset = df[df['Model'] == model].set_index('Term_Clean').reindex(terms)
        
        # Plot points with error bars
        # x=Coef, y=Position
        # error = Coef - Lower, Upper - Coef
        
        y_locs = y_pos + (i * offset * 2) - offset
        
        ax.errorbar(
            x=subset['Coef'],
            y=y_locs,
            xerr=[subset['Coef'] - subset['Lower'], subset['Upper'] - subset['Coef']],
            fmt='o',
            label=model,
            color=colors.get(model, 'black'),
            capsize=5
        )
        
    ax.set_yticks(y_pos)
    ax.set_yticklabels(terms)
    ax.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Effect Coefficient (Std. Units)')
    ax.set_title('Heterogeneity of Evaluation Criteria: Judges vs. Fans (Net Preference)')
    plt.legend(title='Model')
    
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/q3_forest_plot.png', dpi=300)
    print("Saved q3_forest_plot.png")

def plot_variance_decomposition():
    if not os.path.exists(VAR_FILE):
        return
        
    df = pd.read_csv(VAR_FILE)
    
    # Structure: Model, Pro_ICC, Residual
    # We can infer 'Other' (Season/Star/Fixed) = 1 - Pro_ICC - Residual approximately
    # Actually ICC is defined as Var_Group / Total_Random_Var in simpler implementations
    # But usually Total = Group + Residual + Other.
    # Our script calc: Group / (Group + Scale), Residual / (Group + Scale)
    # So they sum to 1.
    
    # Plot Stacked Bar
    df.set_index('Model', inplace=True)
    
    # Ensure columns exist
    if 'Pro_ICC' not in df.columns: return
    
    # Plot
    plt.figure(figsize=(8, 6))
    
    # Reorder for logic
    desired_order = ['Judge', 'Fan_Total', 'Fan_Bias']
    df = df.reindex([m for m in desired_order if m in df.index])
    
    # Columns to plot
    # Pro Effect vs Residual (Context/Idiosyncratic)
    # Multiply by 100 for percentage
    df_pct = df[['Pro_ICC', 'Residual']] * 100
    
    ax = df_pct.plot(kind='bar', stacked=True, color=['#ff7f0e', '#aec7e8'], width=0.6)
    
    plt.title('Influence of Professional Partner (ICC)')
    plt.ylabel('Variance Explained (%)')
    plt.xlabel('Evaluation Metric')
    plt.xticks(rotation=0)
    plt.legend(['Professional Partner', 'Unexplained/Context'], loc='center right')
    
    # Annotate Pro ICC
    for c in ax.containers:
        # Optional: Add labels
        pass
        
    os.makedirs(PLOT_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/q3_variance_plot.png', dpi=300)
    print("Saved q3_variance_plot.png")

if __name__ == "__main__":
    plot_forest_comparison()
    plot_variance_decomposition()
