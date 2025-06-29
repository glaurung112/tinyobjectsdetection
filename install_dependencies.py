import os
import subprocess
import sys

def pip_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

print("Instalando dependencias...\n")

packages = [
    "torch",  # recomendado instalar manualmente con instrucciones específicas si es necesario CUDA
    "torchvision",
    "numpy>=1.22.2",
    "opencv-python>=4.1.1",
    "pillow>=7.1.2",
    "pyyaml>=5.3.1",
    "psutil",
    "matplotlib>=3.3",
    "requests>=2.23.0",
    "scipy>=1.4.1",
    "tqdm>=4.64.0",
    "ultralytics>=8.0.147",
    "thop>=0.1.1",
    "gitpython>=3.1.30",
    "setuptools>=65.5.1",
    "pandas>=1.1.4",
    "seaborn>=0.11.0",
    "scikit-learn<=1.1.2", 
    "kagglehub",
]

for pkg in packages:
    print(f"Instalando {pkg}...")
    pip_install(pkg)

print("\nTodas las dependencias fueron instaladas.")
