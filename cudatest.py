import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")
import torch
print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())
