import sys
import os

print(f"Python Version: {sys.version}")
print(f"Executable: {sys.executable}")

try:
    import torch
    print(f"\n✅ PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("❌ CUDA NOT AVAILABLE (Check your install)")

except ImportError as e:
    print(f"❌ Failed to import torch: {e}")

try:
    import transformers
    print(f"✅ Transformers Version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Failed to import transformers: {e}")

try:
    import peft
    print(f"✅ PEFT Version: {peft.__version__}")
except ImportError as e:
    print(f"❌ Failed to import peft: {e}")

try:
    import accelerate
    print(f"✅ Accelerate Version: {accelerate.__version__}")
except ImportError as e:
    print(f"❌ Failed to import accelerate: {e}")

try:
    import bitsandbytes
    print(f"✅ bitsandbytes Version: {bitsandbytes.__version__}")
except ImportError as e:
    print(f"❌ Failed to import bitsandbytes: {e}")
except Exception as e:
    print(f"⚠️ bitsandbytes imported but threw error (common on Windows): {e}")

print("\n--------------------------")
print("Environment Verification Done")
