import os
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Instalando dependencias...")
print()

# Lista de paquetes
packages = [
    "torch",
    "torchvision",
    "opencv-python",
    "scikit-learn",
    "tqdm",
    "thop",
    "matplotlib",
    "gitpython",
    "numpy",
    "pyyaml",
    "pandas",
    "scipy",
    "requests",
    "pillow",
    "seaborn",
    "kagglehub",
    "setuptools",
    "ultralytics",
    "psutil",
    "tensorboard",
]

for pkg in packages:
    print(f"Instalando {pkg}...")
    pip_install(pkg)