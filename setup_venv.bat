@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    echo Virtual environment already exists: %~dp0.venv
    exit /b 0
)

where python >nul 2>&1
if errorlevel 1 (
    echo Python was not found on PATH.
    exit /b 1
)

echo Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create venv.
    exit /b 1
)

set "VENV_PY=%~dp0.venv\Scripts\python.exe"

echo.
echo Installing NumPy ^(^<2^)...
"%VENV_PY%" -m pip install "numpy<2"
if errorlevel 1 (
    echo NumPy install failed.
    exit /b 1
)

echo.
echo Installing scikit-learn 1.8...
"%VENV_PY%" -m pip install scikit-learn==1.8
if errorlevel 1 (
    echo scikit-learn install failed.
    exit /b 1
)

echo.
echo Installing PyTorch (CUDA 11.8 wheels^)...
"%VENV_PY%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo PyTorch install failed.
    exit /b 1
)



echo.
echo Done. Dependencies are installed in .venv. Activate to use them:
echo   Command Prompt: .venv\Scripts\activate.bat
echo   PowerShell:     .venv\Scripts\Activate.ps1
endlocal
exit /b 0
