"""
download_data.py -- Download all datasets used in the FedSentinel paper.
Paper Table 3: CIFAR-10, CIFAR-100, FEMNIST, Sentiment140
"""

import os, sys, zipfile, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR

def download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"[SKIP] {dest}")
        return
    print(f"[DL] {url}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"[DONE] {dest}")

# CIFAR-10 -- auto via torchvision
import torchvision
print("Downloading CIFAR-10...")
torchvision.datasets.CIFAR10(root=DATA_DIR, download=True)

print("Downloading CIFAR-100...")
torchvision.datasets.CIFAR100(root=DATA_DIR, download=True)

# Sentiment140
print("\nDownloading Sentiment140...")
s140_dir = os.path.join(DATA_DIR, "sentiment140")
os.makedirs(s140_dir, exist_ok=True)
s140_zip = os.path.join(s140_dir, "trainingandtestdata.zip")
download("https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip", s140_zip)
if os.path.exists(s140_zip):
    with zipfile.ZipFile(s140_zip) as zf:
        zf.extractall(s140_dir)
    print("[DONE] Sentiment140 extracted")

# FEMNIST -- manual
print("\nFEMNIST requires LEAF benchmark:")
print("  git clone https://github.com/TalwalkarLab/leaf.git")
print("  cd leaf/data/femnist")
print("  ./preprocess.sh -s niid --sf 0.05 -k 35 -t sample")
print(f"  cp -r data/ {os.path.join(DATA_DIR, 'femnist')}/")

print("\nAll automatic downloads complete.")
