"""
Check that required packages are installed.
"""
import sys

print(f"Python: {sys.version}")
print(f"Path:   {sys.executable}\n")

required = {
    "numpy": "np",
    "torch": "torch",
    "sklearn": "sklearn",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
}

all_ok = True
for name, import_name in required.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"  {name:15s} {ver}")
    except ImportError:
        print(f"  {name:15s} MISSING")
        all_ok = False

print()
if all_ok:
    print("All dependencies installed.")
else:
    print("Missing packages. Run: pip install numpy torch scikit-learn scipy matplotlib")

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU.")
