@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo       VLM Docker Training Setup
echo ==========================================

REM 1. Build the Docker Image
echo [INFO] Building Docker image 'vlm-llava' (This may take a few minutes)...
docker build -t vlm-llava .

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker build failed. Please check if Docker Desktop is running.
    pause
    exit /b %ERRORLEVEL%
)

echo [INFO] Build successful!

REM 2. Run the Container
echo [INFO] Starting Training Container...
echo [INFO] Mounting current directory to /app
echo.

REM Command Explanation:
REM --gpus all: Expose RTX 5070 to container
REM --shm-size 8g: prevent dataloader crashes
REM -v ...: Mount current folder so checkpoints are saved to Windows
docker run --gpus all -it --rm ^
    -e HF_HUB_ENABLE_HF_TRANSFER=0 ^
    -v "%cd%:/app" ^
    --shm-size 8g ^
    vlm-llava python train_mvtec.py

pause
