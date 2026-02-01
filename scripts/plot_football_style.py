import matplotlib.pyplot as plt
import numpy as np

def create_football_style_chart():
    # Data Setup
    rules = [
        'Rank Rule\n(No Save)', 
        'Pct Rule\n(No Save)', 
        'Rank Rule\n(+ Judge Save)'
    ]
    # Reverse for Top-to-Bottom
    rules = rules[::-1]
    
    fan_scores = np.array([0.687, 0.945, 0.680][::-1])
    judge_scores = np.array([0.823, 0.781, 0.835][::-1])
    
    # Layout Config
    # Increased gap significantly to prevent text overlap
    gap = 0.5 
    bar_height = 0.35
    y_pos = np.arange(len(rules))
    
    # Colors
    color_fan = '#66bdce'
    color_judge = '#c1de9c'
    bg_bar_color = '#e0e0e0' 
    alt_band_color = '#fafafa' # Even lighter grey for bands
    text_color_inner = '#333333' 
    
    # Ultra-Wide Figure to maximize visual resolution of differences
    fig, ax = plt.subplots(figsize=(16, 5))
    
    # 1. Row Background Bands
    x_limit = 1.35 # Sufficient to cover bars + labels
    for i in y_pos:
        if i % 2 == 0:
             ax.add_patch(plt.Rectangle((-x_limit, i - 0.5), 2*x_limit, 1.0, 
                                     facecolor=alt_band_color, alpha=1.0, zorder=0))
    
    # 2. "Full Score" Background Bars (Progress Bar Container)
    # Left (Fan)
    ax.barh(y_pos, [1.0]*len(y_pos), height=bar_height, left=-(1.0 + gap/2), 
            color=bg_bar_color, edgecolor='none', zorder=1)
            
    # Right (Judge)
    ax.barh(y_pos, [1.0]*len(y_pos), height=bar_height, left=gap/2, 
            color=bg_bar_color, edgecolor='none', zorder=1)

    # 3. Actual Score Bars'
    # Left (Fan)
    ax.barh(y_pos, fan_scores, height=bar_height, left=-(fan_scores + gap/2), 
            color=color_fan, zorder=3)
            
    # Right (Judge)
    ax.barh(y_pos, judge_scores, height=bar_height, left=gap/2, 
            color=color_judge, zorder=3)
    
    # 4. Center Text (Rule Names)
    for i, rule in enumerate(rules):
        # Increased font size slightly since we have space
        ax.text(0, i, rule, ha='center', va='center', fontsize=11, 
                fontweight='bold', color='#222222', zorder=5)

    # 5. Value Labels
    def add_value_labels(scores, is_left):
        for i, score in enumerate(scores):
            label = f"{score:.3f}"
            
            # Position
            if is_left:
                x_text = -(score + gap/2) + 0.05 
                ha_align = 'left'
            else:
                x_text = (score + gap/2) - 0.05
                ha_align = 'right'
            
            ax.text(x_text, i, label, ha=ha_align, va='center', 
                    color=text_color_inner, fontweight='bold', fontsize=11, zorder=10)

    add_value_labels(fan_scores, True)
    add_value_labels(judge_scores, False)

    # 6. Headers
    header_y = len(rules) - 0.35
    # Adjusted header position based on new gap
    ax.text(-(gap/2 + 0.5), header_y, "Fan Alignment", ha='center', va='center', 
            fontsize=15, fontweight='bold', color='black') 
    ax.text((gap/2 + 0.5), header_y, "Judge Alignment", ha='center', va='center', 
            fontsize=15, fontweight='bold', color='black') 

    # Clean Axes
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.5, len(rules))
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('output/rule_comparison_football_style.png', dpi=300, bbox_inches='tight')
    print("Chart saved to output/rule_comparison_football_style.png")

if __name__ == "__main__":
    create_football_style_chart()
