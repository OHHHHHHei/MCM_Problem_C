# 2026 MCM Problem C: "Unraveling the Votes"
## SMC-Inverse Latent State Reconstruction System

> A Sequential Monte Carlo (Particle Filter) approach to reconstruct hidden fan voting shares in *Dancing with the Stars*, based on observed elimination events and judge scores.

---

### 🔥 Key Features
*   **Bayesian Reconstruction**: Uses SMC (Particle Filtering) to invert the "Black Box" of elimination results, estimating the latent probability of fan support for every couple in every week.
*   **Dynamic Modeling**: Captures "Hometown Glory" vs "What have you done for me lately?" using a dual-state model (Long-term Popularity $\mu$ + Short-term Momentum $x$).
*   **Mechanism-Agnostic**: Handles complex rule changes across 33 seasons (Judge Saves, Double Eliminations, Rank vs Percentage rules).
*   **Optimized Parameters**: Calibrated on 30+ seasons of historical data to maximize predictive accuracy (Top-3 Hit Rate ~80%).

---

### 📂 Directory Structure

The project has been refactored for modularity and scalability:

```text
Code/
├── main.py                     # Main Entry Point (Run this!)
├── 2026_MCM_Problem_C_Data.csv # Source Data
├── core/                       # [Core Package]
│   ├── smc_inverse.py          # Particle Filter Engine (SMC)
│   ├── data_processor.py       # Data Loading & Feature Engineering
│   ├── competition_rules.py    # Rule Set Engine (Elimination Logic)
│   └── analysis.py             # Result Analyzer & Metrics
├── plots/                      # [Visualization]
│   ├── plot_consistency_check.py     # Calibration Curves
│   ├── plot_forensic_analysis.py     # Case Studies (e.g., Bobby Bones)
│   ├── plot_memory_goldilocks.py     # Parameter Sensitivity
│   └── ...
├── optimization/               # [Hyperparameter Tuning]
│   ├── optimizer_overnight.py  # Robust Grid Search
│   └── optimizer_parallel.py   # Aggressive Parallel Search
├── scripts/                    # [Utilities]
│   └── extract_highlights.py   # Extract best models from logs
└── tests/                      # [Unit Tests]
    └── test_debug.py           # Integrity Check
```

---

### 🚀 Quick Start

#### 1. Setup Environment
Ensure you have Python 3.8+ and standard scientific libraries installed.
```bash
# Install dependencies (Standard Stack)
pip install numpy pandas scipy matplotlib seaborn tqdm
```

#### 2. Run the Reconstruction Model
The `main.py` script runs the model using the **Optimized Parameters**.

```bash
# Run full analysis on all seasons (Recommended)
python main.py

# Run in test mode (fewer particles, faster)
python main.py --test

# Run specific seasons
python main.py --seasons 27 19 31
```

#### 3. Visualize Results
Generate academic-quality plots in the `output/` directory:

```bash
# Example: Generate Validation Consistency Plot
python plots/plot_consistency_check.py

# Example: Forensic Analysis of "The Bobby Bones Effect" (Season 27)
python plots/plot_forensic_analysis.py
```

---

### 🧠 Model Parameters (Optimized)

Based on our extensive grid search (Grid Size: ~5000 combinations), the model defaults to the **"Overall Best"** configuration:

| Parameter | Symbol | Value | Interpretation |
| :--- | :---: | :---: | :--- |
| **Memory Decay** | $\rho$ | **0.9** | **Elephants never forget.** Fans have long memories; accumulated popularity dominates temporary performance spikes. |
| **Judge Influence** | $\gamma$ | **0.0** | **Independence.** Fan voting behavior is effectively independent of judge scoring cues. |
| **Shock Threshold** | $\delta$ | **1.2** | **Stability.** Only massive performance deviations trigger structural popularity shifts. |
| **Discriminant** | $\alpha$ | **7.0** | **Competition.** The voting distribution is sharp; the gap between top and bottom is significant. |

*(You can override these defaults via command line, e.g., `python main.py --rho 0.5 --gamma 0.5`)*

---

### 📊 Performance Metrics

*   **Top-3 Hit Rate**: ~79.2% (The model correctly identifies the actual eliminated contestant in its top-3 risk list 4 out of 5 times).
*   **MAP Match Rate**: ~82.2% (The model's "Most Likely" outcome matches the deterministic elimination rules 82% of the time).
*   **Prediction**: Strictly "Pre-update" (No data leakage).
*   **Reconstruction**: "Post-update" (Incorporates all history).

---

### 📝 Usage Notes

*   **Output**: All results (JSON/TXT/PNG) are saved to the `output/` folder.
*   **Logs**: Check `analysis_report.txt` for a detailed breakdown of every season processed.
*   **Debugging**: Run `python tests/test_debug.py` to verify your environment path mapping.

---
*MCM 2026 Problem C Team*
