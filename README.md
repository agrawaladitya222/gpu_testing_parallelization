# GPU testing / parallelization

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
