@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   VAEpp0r Installer
echo ============================================
echo.

:: ---- Check for Python ----
set "PYTHON="

:: Try py launcher first (most reliable on Windows)
where py >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('py -3 --version 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PYTHON=py -3"
        echo Found !PY_VER! via py launcher
        goto :python_ok
    )
)

:: Try python on PATH
where python >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%i in ('python --version 2^>nul') do set "PY_VER=%%i"
    if defined PY_VER (
        set "PYTHON=python"
        echo Found !PY_VER!
        goto :python_ok
    )
)

:: Try common install locations
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if exist %%P (
        for /f "tokens=*" %%i in ('%%P --version 2^>nul') do set "PY_VER=%%i"
        if defined PY_VER (
            set "PYTHON=%%~P"
            echo Found !PY_VER! at %%P
            goto :python_ok
        )
    )
)

:: Python not found — try to install
echo Python not found. Attempting to install...
echo.

:: Try winget first
where winget >nul 2>&1
if %errorlevel%==0 (
    echo Installing Python 3.12 via winget...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if %errorlevel%==0 (
        echo.
        echo Python installed. Restarting installer to pick up new PATH...
        echo Please close this window and run install.bat again.
        pause
        exit /b 0
    )
)

:: Manual download fallback
echo.
echo Could not install Python automatically.
echo Please install Python 3.10+ from https://www.python.org/downloads/
echo IMPORTANT: Check "Add Python to PATH" during installation.
echo Then run this installer again.
pause
exit /b 1

:python_ok
echo.

:: ---- Check Python version >= 3.10 ----
for /f "tokens=2 delims= " %%v in ("!PY_VER!") do set "VER=%%v"
for /f "tokens=1,2 delims=." %%a in ("!VER!") do (
    set "MAJOR=%%a"
    set "MINOR=%%b"
)
if !MAJOR! LSS 3 (
    echo ERROR: Python 3.10+ required, found !PY_VER!
    pause
    exit /b 1
)
if !MAJOR!==3 if !MINOR! LSS 10 (
    echo ERROR: Python 3.10+ required, found !PY_VER!
    pause
    exit /b 1
)

:: ---- Create venv ----
if exist venv (
    echo Virtual environment already exists.
) else (
    echo Creating virtual environment...
    !PYTHON! -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create venv.
        pause
        exit /b 1
    )
    echo Done.
)
echo.

:: ---- Detect GPU ----
set "HAS_NVIDIA=0"
where nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    nvidia-smi >nul 2>&1
    if %errorlevel%==0 (
        set "HAS_NVIDIA=1"
        echo NVIDIA GPU detected.
        for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2^>nul') do (
            echo   GPU: %%i
        )
    )
)

:: ---- Install PyTorch ----
echo.
if !HAS_NVIDIA!==1 (
    echo Installing PyTorch with CUDA 12.4...
    venv\Scripts\pip install --upgrade pip >nul 2>&1
    venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
) else (
    echo No NVIDIA GPU detected. Installing CPU-only PyTorch...
    echo (Training will be slow without a GPU^)
    venv\Scripts\pip install --upgrade pip >nul 2>&1
    venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

if %errorlevel% neq 0 (
    echo ERROR: PyTorch installation failed.
    pause
    exit /b 1
)

:: ---- Install requirements ----
echo.
echo Installing dependencies...
venv\Scripts\pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

:: ---- Verify ----
echo.
echo Verifying installation...
venv\Scripts\python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); [print(f'  {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
if %errorlevel% neq 0 (
    echo WARNING: Verification failed, but installation may still work.
)

:: ---- Done ----
echo.
echo ============================================
echo   Installation complete!
echo ============================================
echo.
echo   Run gui.bat to launch the GUI.
echo   Or use the training scripts directly:
echo     venv\Scripts\python -m training.train_static --help
echo     venv\Scripts\python -m training.train_video --help
echo.
pause
