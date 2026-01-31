import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"Failed to import numpy: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except ImportError as e:
    print(f"Failed to import pandas: {e}")

try:
    import scipy
    print(f"SciPy version: {scipy.__version__}")
except ImportError as e:
    print(f"Failed to import scipy: {e}")

try:
    from core.smc_inverse import SMCInverse, ModelParams
    from core.data_processor import DataProcessor
    print("Successfully imported project modules.")
    
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../2026_MCM_Problem_C_Data.csv'))
    dp = DataProcessor(csv_path)
    print("Successfully initialized DataProcessor.")
    
    model = SMCInverse(dp, ModelParams(n_particles=10))
    print("Successfully initialized SMCInverse.")
    
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
