import os
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Instalando dependencias...")
print()

# Instalar dependencias de yolov5 en yolov5/requirements.txt si existe
if os.path.exists("yolov5-cbam/requirements.txt"):
    print("Instalando desde requirements.txt...")
    pip_install("-r yolov5-cbam/requirements.txt")

# Lista de paquetes
packages = [
    "torch",
    "opencv-python",
    "scikit-learn",
    "tqdm",
    "matplotlib",
    "pyyaml",
    "pandas",
    "requests",
    "pillow",
    "seaborn",
    "kagglehub",
]

for pkg in packages:
    print(f"Instalando {pkg}...")
    pip_install(pkg)