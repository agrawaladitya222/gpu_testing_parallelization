# GPU testing / parallelization

## Virtual environment

Use a local environment in `.venv` under this project directory so CUDA wheels and version pins do not affect your global Python.

**Create** (from this folder, in PowerShell or Command Prompt):

```powershell
python -m venv .venv
```

**Or** double-click or run `setup_venv.bat` in this folder; it creates `.venv` and, on a fresh create only, runs the same `pip install` commands as in **Dependencies** below. If `.venv` already exists, the script exits without reinstalling packages.

**Activate** (do this before running your code or any extra `pip install`):

- **PowerShell:** `.venv\Scripts\Activate.ps1`
- **Command Prompt:** `.venv\Scripts\activate.bat`

If PowerShell blocks the script, use Command Prompt activation or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Dependencies (old Windows + GPU)

Specific environment required for GPU use on an older Windows setup.

**PyTorch (CUDA 11.8 wheels):**

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**NumPy (stay below 2.x for compatibility):**

```powershell
pip install "numpy<2"
```

**scikit-learn 1.8:**

```powershell
pip install scikit-learn==1.8
```
