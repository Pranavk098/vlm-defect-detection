@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       VLM Project Environment Setup
echo ==========================================

REM 1. Create venv if it doesn't exist
if not exist "venv_vlm" (
    echo [INFO] Creating virtual environment 'venv_vlm'...
    python -m venv venv_vlm
) else (
    echo [INFO] 'venv_vlm' already exists.
)

REM 2. Activate venv
call venv_vlm\Scripts\activate.bat

REM 3. Upgrade pip
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip

REM 4. Install dependencies
echo [INFO] Installing requirements from requirements.txt...
pip install -r requirements.txt

REM 5. Install LLaVA from source (Editable mode)
if exist "LLaVA" (
    echo [INFO] Installing LLaVA package in editable mode...
    cd LLaVA
    pip install -e .
    cd ..
) else (
    echo [WARNING] LLaVA directory not found! Skipping LLaVA installation.
    echo Please make sure you clone LLaVA into this folder.
)

REM 6. Verify Installation
echo [INFO] Running verify_install.py...
python verify_install.py

echo.
echo ==========================================
echo       Setup Complete!
echo ==========================================
echo To start using the environment, run: venv_vlm\Scripts\activate.bat
pause
