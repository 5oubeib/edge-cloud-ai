import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt

print("✓ PyTorch version:", torch.__version__)
print("✓ TorchVision version:", torchvision.__version__)
print("✓ Pandas version:", pd.__version__)
print("✓ CUDA available:", torch.cuda.is_available())
print("\n All dependencies installed successfully!")
