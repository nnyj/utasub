import os

# bit-stable FA output across sessions regardless of GPU state; must precede
# cuBLAS handle creation (first CUDA use), hence set at package import time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
