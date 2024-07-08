import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "torch",
    "torchvision",
    "torch.optim",
    "sklearn",
    "seaborn",
    "tqdm",
    "librosa",
    "pydub",
    "soundfile",
    "pillow",
    "ffmpeg",
]

for package in packages:
    install(package)
