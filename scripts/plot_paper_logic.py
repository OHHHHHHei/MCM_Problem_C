import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import textwrap
import numpy as np
import os

def draw_complex_flowchart():
    # Setup Figure (16:9 aspect ratio, high resolution)
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # --- Style Constants ---
    COLOR_DATA = '#E1F5FE'      # Light Blue (Data)
    COLOR_MODEL = '#FFF3E0'     # Light Orange (Model)
    COLOR_ANALYSIS = '#F3E5F5'  # Light Purple (Analysis)
    COLOR_OPTIM = '#E8F5E9'     # Light Green (Optimization)
    COLOR_EDGE = '#455A64'
    FONT_TITLE = {'family': 'sans-serif', 'weight': 'bold', 'size': 12}
    FONT_BODY = {'family': 'sans-serif', 'size': 9}
    
    # --- Helpers ---
    def add_box(x, y, w, h, title, content, color, edge=COLOR_EDGE):
        # Shadow
        shadow = patches.FancyBboxPatch((x+0.1, y-0.1), w, h, boxstyle="round,pad=0.1", 
                                        linewidth=0, facecolor='#DDDDDD', zorder=1)
        ax.add_patch(shadow)
        
        # Box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=1.5, edgecolor=edge, facecolor=color, zorder=2)
        ax.add_patch(rect)
        
        # Title
        ax.text(x + 0.2, y + h - 0.4, title, ha='left', va='top', **FONT_TITLE, color='#263238', zorder=3)
        
        # Content (Wrapped)
        wrapped_text = textwrap.fill(content, width=int(w * 8)) # Approx char width
        ax.text(x + w/2, y + h/2 - 0.2, wrapped_text, ha='center', va='center', **FONT_BODY, color='#37474F', zorder=3)
        return rect

    def add_arrow(x1, y1, x2, y2, style="->", color=COLOR_EDGE, curved=False):
        con = f"arc3,rad=0.2" if curved else "arc3,rad=0"
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, connectionstyle=con, color=color, lw=1.5, mutation_scale=15), zorder=1)

    # ==========================
    # 1. INPUT STAGE (Left)
    # ==========================
    add_box(0.5, 7.0, 3.0, 2.0, "Input Data", 
            "Historical Elimination Data (S1-S34)\nJudge Scores Matrix\nSocial Media Proxies", 
            COLOR_DATA)

    # ==========================
    # 2. Q1: RECONSTRUCTION (Top-Left Center)
    # ==========================
    # Container
    ax.add_patch(patches.Rectangle((4.0, 6.5), 4.5, 3.0, linewidth=1, edgecolor='#B0BEC5', facecolor='none', linestyle='--', zorder=0))
    ax.text(4.0, 9.6, "Module 1: SMC Reconstruction", fontsize=10, weight='bold', color='#78909C')
    
    # Sub-components
    add_box(4.2, 7.5, 1.8, 1.2, "SMC-Inverse", "Particle Filtering\nSequential Monte Carlo", COLOR_MODEL)
    add_box(6.5, 7.5, 1.8, 1.2, "Latent State", "Unobserved Vote Shares\n(Posterior Dist)", COLOR_DATA)
    
    # Internal Arrow
    add_arrow(6.0, 8.1, 6.5, 8.1)

    # ==========================
    # 3. Q2: DIAGNOSIS (Top-Right Center)
    # ==========================
    # Container
    ax.add_patch(patches.Rectangle((9.0, 6.5), 4.5, 3.0, linewidth=1, edgecolor='#B0BEC5', facecolor='none', linestyle='--', zorder=0))
    ax.text(9.0, 9.6, "Module 2: Counterfactual Audit", fontsize=10, weight='bold', color='#78909C')
    
    add_box(9.2, 7.5, 1.8, 1.2, "Simulation", "N=100 Monte Carlo\nRule Variants", COLOR_MODEL)
    add_box(11.5, 7.5, 1.8, 1.2, "Discrepancy", "Metric: Alignment Gap\nResult: 'Magnitude Paradox'", COLOR_ANALYSIS)
    
    add_arrow(11.0, 8.1, 11.5, 8.1)

    # ==========================
    # 4. Q3: CONTEXT (Bottom-Left)
    # ==========================
    add_box(4.0, 2.5, 4.5, 2.0, "Module 3: Heterogeneity", 
            "Regression Analysis (S1/S2 Scores)\nClustering: Populists vs Tech-Victims", 
            COLOR_ANALYSIS)

    # ==========================
    # 5. Q4: DESIGN (Bottom-Right)
    # ==========================
    # Container
    ax.add_patch(patches.Rectangle((9.0, 1.0), 6.5, 4.0, linewidth=1, edgecolor='#B0BEC5', facecolor='none', linestyle='--', zorder=0))
    ax.text(9.0, 5.1, "Module 4: ACDW Mechanism Design", fontsize=10, weight='bold', color='#78909C')
    
    # Components representing formula parts
    add_box(9.5, 2.5, 2.0, 1.5, "Concave Utility", "u(v) = v^p\n(p=0.55)\nSoft Cap Logic", COLOR_MODEL)
    add_box(12.5, 2.5, 2.0, 1.5, "Dynamic Weight", "λ = f(ρ)\nJudge Agency", COLOR_MODEL)
    
    # Add a small visual curve for Concave Utility
    x = np.linspace(9.6, 11.4, 50)
    y = 0.5 * (x - 9.6)**0.5 + 2.8 # Fake curve
    # ax.plot(x, y, color='#D81B60', lw=2, zorder=4) # Visual embellishment

    # ==========================
    # 6. OUTPUT (Far Right)
    # ==========================
    add_box(16.5, 4.0, 3.0, 2.0, "Final Policy", 
            "ACDW-B3 System\n\nOptimal Pareto Frontier\nScore: 0.905", 
            COLOR_OPTIM)

    # ==========================
    # LINKS (The Flows)
    # ==========================
    
    # Input -> Q1
    add_arrow(3.5, 8.0, 4.2, 8.1)
    
    # Q1 -> Q2 (Data feeding Simulation)
    add_arrow(8.3, 8.1, 9.2, 8.1, color='#0288D1') # Blue for Data Flow
    
    # Q1 -> Q3 (Data feeding Analysis)
    add_arrow(6.4, 7.5, 6.25, 4.5, style="->", curved=False, color='#0288D1')
    
    # Q3 -> Q4 (Context informing Design)
    add_arrow(8.5, 3.5, 9.5, 3.25, curved=True, color=COLOR_EDGE)
    
    # Q2 -> Q4 (Problem informing Solution)
    add_arrow(12.4, 7.5, 13.5, 4.0, curved=True, color=COLOR_EDGE)
    
    # Q4 Internal Synthesis
    # Concave + Dynamic -> System
    
    # Q4 -> Output
    add_arrow(15.5, 3.25, 16.5, 5.0)

    # ==========================
    # ANNOTATIONS
    # ==========================
    ax.text(10, 9.5, "Figure 1: The 'Reconstruction-Diagnosis-Optimization' Framework", 
            ha='center', fontsize=16, weight='bold', color='#455A64')
            
    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images', 'paper_logic_chain_complex.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Comparison Flowchart saved to {output_path}")

if __name__ == "__main__":
    draw_complex_flowchart()
