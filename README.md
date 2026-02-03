
# 2026 MCM Problem C: Deciphering the Invisible Vote
### Team #2607504 Solution Repository

![MCM Problem C](https://img.shields.io/badge/MCM-Problem%20C-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![Status](https://img.shields.io/badge/Status-Complete-success)

> A comprehensive data science framework to reconstruct latent fan voting shares, analyze rule fairness, and design optimal scoring mechanisms for *Dancing with the Stars*.

---

## 📄 Abstract

This project addresses the "Black Box" of fan voting in *Dancing with the Stars* using a four-stage modeling approach:
1.  **Latent State Reconstruction (SMC)**: Using **Particle Filtering** to invert elimination results and estimate weekly fan vote shares for 34 seasons.
2.  **Counterfactual Analysis**: Simulating "What If" scenarios to compare **Rank Rule vs. Percentage Rule**, revealing bias ($S_1$) and controversy ($S_2$) metrics.
3.  **Factor Quantification (3T-MEM)**: A **Triple-Track Linear Mixed-Effects Model** to disentangle the impact of technical skills, celebrity stardom, and professional partners.
4.  **System Design (ACDW-B3)**: Proposing the **Adaptive Concave Diminishing-returns Weighted Bottom-3** system to maximize both fairness and entertainment value.

---

## 🚀 Key Modules & Usage

The project is organized by the four main tasks of the competition.

### 1. Latent Vote Reconstruction (Task 1)
Reconstructs the hidden fan vote shares ($\pi_{i,t}$) using Sequential Monte Carlo.
*   **Core Script**: `main.py`
*   **Algorithm**: Particle Filter (SMC) with dual-state momentum (Long-term $\mu$ + Short-term $x$).
*   **Usage**:
    ```bash
    # Run reconstruction for all seasons
    python main.py
    ```

### 2. Controversy & Rule Analysis (Task 2)
Analyzes "Robbed" contestants and compares Rank vs. Percentage rules.
*   **Core Script**: `scripts/analyze_controversy_v3.py`
*   **Feature**: Calculates $S_1$ (Fan-Carried) and $S_2$ (Robbed) scores for all contestants.
*   **Usage**:
    ```bash
    # Generate controversy report
    python scripts/analyze_controversy_v3.py
    
    # Run counterfactual simulations (e.g., "What if Bobby Bones faced the Rank Rule?")
    python scripts/analyze_case_counterfactuals.py
    ```

### 3. Factor Analysis (Task 3)
Quantifies the impact of Industry, Age, and Partners.
*   **Core Script**: `scripts/analyze_q3_3t_mem.py`
*   **Feature**: Runs the 3T-MEM model to calculate ICC for partners and coefficients for industries.
*   **Visualization**: `scripts/plot_industry_balloon.py` (Generates the Balloon Plot).
*   **Usage**:
    ```bash
    python scripts/analyze_q3_3t_mem.py
    ```

### 4. Mechanism Design (Task 4)
Simulates the new **ACDW-B3** system.
*   **Core Script**: `scripts/benchmark_acdw_full.py`
*   **Feature**: Compares ACDW-B3 against Rank and Percentage rules on Fairness and Consensus metrics.
*   **Usage**:
    ```bash
    python scripts/benchmark_acdw_full.py
    ```

---

## 📂 Directory Structure

```text
Code/
├── main.py                     # [Task 1] Primary entry point for SMC Reconstruction
├── docs/                       # Project Documentation
│   ├── 终稿.pdf              # Final Submission Paper
│   ├── Memo_Final.tex          # Final Recommendation Memo (LaTeX)
│   └── ...
├── core/                       # Core Algorithms
│   ├── smc_inverse.py          # Particle Filter Implementation
│   └── competition_rules.py    # Historical Rules Engine
├── scripts/                    # Analysis Suites
│   ├── analyze_controversy_v3.py   # [Task 2] Controversy Metrics
│   ├── analyze_q3_3t_mem.py        # [Task 3] Mixed-Effects Model
│   ├── benchmark_acdw_full.py      # [Task 4] New System Simulation
│   └── plot_industry_balloon.py    # Visualization Scripts
├── output/                     # Generated results (logs, plots, json)
└── data/                       # Dataset (2026_MCM_Problem_C_Data.csv)
```

---

## 🏆 Key Findings

*   **Pro-Fan Bias**: The **Percentage Rule** has a 0.945 bias towards popularity, often allowing superstars to bypass technical requirements.
*   **The "Partner Myth"**: Using LMM, we proved that **Professional Partners** have a negligible statistical impact on outcomes compared to Celebrity Industry/Fame.
*   **Optimal System**: The proposed **ACDW-B3** system achieves **97.2% Judge Alignment** while preserving fan agency, effectively solving the "Bobby Bones Problem."

---

## 🛠 Prerequisites

*   Python 3.8+
*   Packages: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `statsmodels`

```bash
pip install -r requirements.txt
```

---
*© 2026 MCM Team #2607504*
