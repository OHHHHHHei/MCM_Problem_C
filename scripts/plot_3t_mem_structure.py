import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import os

def draw_3t_mem_diagram():
    # Figure Setup (Landscape 16:9)
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # --- Styles ---
    # Colors
    C_DATA = '#E3F2FD'    # Blue
    C_CORE = '#FFF3E0'    # Orange
    C_VAR = '#E8F5E9'     # Green
    C_RAND = '#F3E5F5'    # Purple
    C_OUT = '#FFFDE7'     # Yellow
    C_EDGE = '#546E7A'
    
    # Fonts
    F_TITLE = {'weight': 'bold', 'size': 12, 'ha': 'left'}
    F_BODY = {'size': 10, 'ha': 'center', 'va': 'center'}
    
    # --- Helpers ---
    def add_box(x, y, w, h, text, color, label=None, style='solid'):
        # Main Box
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                                      linewidth=1.5, edgecolor=C_EDGE, facecolor=color, linestyle=style)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, **F_BODY)
        
        # Label (e.g. "Track A")
        if label:
            ax.text(x, y + h + 0.1, label, size=9, weight='bold', color='#455A64')
            
        return rect
        
    def add_container(x, y, w, h, title):
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='#CFD8DC', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x + 0.2, y + h - 0.4, title, **F_TITLE, color='#90A4AE')

    def add_arrow(x1, y1, x2, y2, style='->', ls='-'):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=C_EDGE, lw=1.5, ls=ls, mutation_scale=15))

    # ==============================
    # 1. LAYOUT ZONES
    # ==============================
    # Coordinates (approx grid)
    # X: Input(1-4), Vars(5-7), Core(8-13), Output(14-16)
    # Y: Top(8), Mid(5), Bot(2)
    
    # Containers
    add_container(0.5, 0.5, 4.0, 9.0, "Data Input")
    add_container(5.0, 0.5, 2.5, 9.0, "Explanatory & Random")
    add_container(8.0, 0.5, 6.0, 9.0, "3T-MEM Core Models")
    
    # ==============================
    # 2. DATA INPUT (Left)
    # ==============================
    # Raw -> YJ
    add_box(1.0, 7.0, 3.0, 1.2, "Raw Judge Scores", C_DATA)
    add_arrow(2.5, 7.0, 2.5, 6.2)
    add_box(1.0, 5.0, 3.0, 1.2, "Standardization\n(Z-Score)", C_DATA) 
    # Use YJ variable
    rect_yj = add_box(1.0, 3.0, 3.0, 1.2, "YJ: Norm. Judge Score", C_DATA, label="Dependent Var A")
    add_arrow(2.5, 5.0, 2.5, 4.2)
    
    # SMC -> YF
    add_box(1.0, 2.0, 3.0, 0.8, "SMC Inverse\n(Latent Votes)", C_DATA) # Squeezed a bit low? Move up logic? 
    # Actually diagram text says SMC -> Logit -> YF. 
    # Let's adjust positions.
    
    # Re-plan Data Input positions
    # Top: Judge
    add_box(1.2, 8.0, 2.6, 0.8, "Raw Data", C_DATA)
    add_arrow(2.5, 8.0, 2.5, 7.0)
    add_box(1.2, 5.8, 2.6, 1.2, "YJ (Z-Score)", C_DATA, label="Metric A")
    
    # Bottom: Fan
    add_box(1.2, 2.5, 2.6, 0.8, "SMC Output", C_DATA)
    add_arrow(2.5, 2.5, 2.5, 3.5)
    add_box(1.2, 3.8, 2.6, 1.2, "YF (Logit Share)", C_DATA, label="Metric B")

    # ==============================
    # 3. VARIABLES (Middle Left)
    # ==============================
    # Explanatory
    add_box(5.2, 6.5, 2.1, 1.5, "Fixed Effects\n(Traits)\nAge, Industry", C_VAR)
    
    # Random
    add_box(5.2, 2.5, 2.1, 1.5, "Random Effects\nPro ID\nStar ID", C_RAND)

    # ==============================
    # 4. CORE MODELS (Center/Right)
    # ==============================
    # Track A
    add_box(8.5, 7.0, 5.0, 2.0, "Track A: Judge Elite Model\nYJ ~ Traits + (1|Pro) + (1|Star)", C_CORE)
    
    # Track B1
    add_box(8.5, 4.0, 5.0, 2.0, "Track B1: Fan Popularity\nYF ~ Traits + (1|Pro) + (1|Star)", C_CORE)
    
    # Track B2
    add_box(8.5, 1.0, 5.0, 2.0, "Track B2: Net Preference\nYF ~ YJ + Traits + ...", C_CORE)

    # ==============================
    # 5. OUTPUT (Far Right)
    # ==============================
    add_box(14.5, 3.5, 3.0, 3.5, "Heterogeneity\nAnalysis\n\nCompare Coeffs:\nBeta(Judge)\nvs\nBeta(Fan)", C_OUT)

    # ==============================
    # 6. LINKS
    # ==============================
    
    # Data to Models
    # YJ -> Track A
    add_arrow(3.8, 6.4, 8.5, 8.0, ls='-') 
    # YF -> Track B1
    add_arrow(3.8, 4.4, 8.5, 5.0, ls='-')
    # YF -> Track B2
    add_arrow(3.8, 4.4, 8.5, 2.0, ls='-')
    # YJ -> Track B2 (Control)
    add_arrow(3.8, 6.4, 8.5, 2.5, ls='--') # Control variable dashed?

    # Traits to Models
    # Traits -> A, B1, B2
    add_arrow(7.3, 7.25, 8.5, 7.8)
    add_arrow(7.3, 7.25, 8.5, 5.2)
    add_arrow(7.3, 7.25, 8.5, 2.2)
    
    # Random to Models
    add_arrow(7.3, 3.25, 8.5, 2.1, ls=':') # Dotted for Random
    add_arrow(7.3, 3.25, 8.5, 4.8, ls=':')
    add_arrow(7.3, 3.25, 8.5, 7.6, ls=':')

    # Models to Heterogeneity
    add_arrow(13.5, 8.0, 15.0, 7.0) # From A
    add_arrow(13.5, 2.0, 15.0, 3.5) # From B2 (Comparison Target)

    # ==============================
    # TITLE
    # ==============================
    ax.text(9.0, 9.6, "The 3T-MEM (Three-Track Mixed Effects Model) Framework", 
            ha='center', fontsize=16, weight='bold', color='#263238')

    # Save
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'images', '3t_mem_framework.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Diagram saved to {output_path}")

if __name__ == "__main__":
    draw_3t_mem_diagram()
