@echo off
setlocal
title LLaVA Environment Setup

echo ===================================================
echo       LLaVA Local Windows Setup (RTX 5070)
echo ===================================================

:: 1. Check Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH. Please install Python 3.10 or newer.
    pause
    exit /b 1
)
python --version

:: 2. Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment 'venv'...
    python -m venv venv
) else (
    echo [INFO] 'venv' already exists. skipping creation.
)

:: 3. Activate Environment
echo [INFO] Activating venv...
call venv\Scripts\activate.bat

:: 4. Clone LLaVA
if not exist "LLaVA" (
    echo [INFO] Cloning LLaVA repository...
    git clone https://github.com/haotian-liu/LLaVA.git
) else (
    echo [INFO] LLaVA directory found.
)

:: 5. Install Dependencies
echo [INFO] Installing Dependencies from requirements.txt...
echo        (This may take a while for Torch download)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

:: 6. Install LLaVA in Editable Mode
echo [INFO] Installing LLaVA...
cd LLaVA
:: Use --no-deps to avoid overriding our carefully selected versions
pip install -e . --no-deps
cd ..

echo ===================================================
echo [SUCCESS] Environment Setup Complete!
echo.
echo To run training:
echo    1. venv\Scripts\activate
echo    2. python train_mvtec.py
echo ===================================================
pause
