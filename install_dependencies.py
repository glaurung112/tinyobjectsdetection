import os
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Instalando dependencias...")

# Instalar dependencias de yolov5 en YOLO/yolov5/requirements.txt si existe
if os.path.exists("requirements.txt"):
    print("📦 Instalando desde requirements.txt...")
    pip_install("-r requirements.txt")

# Lista de paquetes manuales
packages = [
    "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
    "opencv-python",
    "scikit-learn",
    "tqdm",
    "matplotlib",
    "pyyaml",
    "pandas",
    "requests",
    "pillow",
    "seaborn",
]

for pkg in packages:
    print(f"📦 Instalando {pkg}...")
    pip_install(pkg)

# Comprobar disponibilidad de GPU
import torch
print(f"✅ PyTorch instalado correctamente. GPU disponible: {torch.cuda.is_available()}")
