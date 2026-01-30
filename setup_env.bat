@echo off
REM SMC-Inverse 环境配置脚本
REM 使用方法: 双击运行此脚本或在命令行执行 setup_env.bat

echo ========================================
echo SMC-Inverse Environment Setup
echo ========================================

REM 检查 conda 是否可用
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Conda not found! Please install Anaconda or Miniconda first.
    echo Download from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo.
echo [Step 1] Creating conda environment 'smc_inverse'...
conda create -n smc_inverse python=3.10 -y

echo.
echo [Step 2] Activating environment...
call conda activate smc_inverse

echo.
echo [Step 3] Installing dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment, run:
echo   conda activate smc_inverse
echo.
echo To run the model:
echo   python main.py --test        (quick test)
echo   python main.py               (full run)
echo   python main.py --help        (show options)
echo.
pause
