# =============================================================================
# dataset.py -- Dataset loading, non-IID partitioning, and DataLoaders
# Paper Section 6.1 -- Datasets and non-IID Dirichlet partitioning (α_Dir=0.5)
# Datasets: CIFAR-10, CIFAR-100, FEMNIST, Sentiment140
# =============================================================================

import os, json, csv, zipfile, urllib.request, tarfile
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T

from config import (DATA_DIR, DATASET, N_CLIENTS, DIRICHLET_ALPHA,
                    BATCH_SIZE, NUM_WORKERS, SEED,
                    DATASET_CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Download helpers
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[dataset] Already exists: {dest}")
        return
    print(f"[dataset] Downloading {url} -> {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest)

# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 / CIFAR-100  (torchvision)
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar_transforms(dataset: str = "cifar10") -> Tuple:
    """Standard CIFAR augmentation pipeline."""
    # Section 6 -- standard normalization; no specific augmentation listed
    mean = (0.4914, 0.4822, 0.4465) if dataset == "cifar10" else (0.5071, 0.4867, 0.4408)
    std  = (0.2470, 0.2435, 0.2616) if dataset == "cifar10" else (0.2675, 0.2565, 0.2761)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, test_tf


def load_cifar(dataset: str = "cifar10") -> Tuple[Dataset, Dataset]:
    """Download and load CIFAR-10 or CIFAR-100 via torchvision.
    Download URL (auto): https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    cfg = DATASET_CONFIG[dataset]
    train_tf, test_tf = get_cifar_transforms(dataset)
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    train_ds = cls(root=DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds  = cls(root=DATA_DIR, train=False, download=True, transform=test_tf)
    return train_ds, test_ds

# ─────────────────────────────────────────────────────────────────────────────
# FEMNIST  (LEAF benchmark)
# ─────────────────────────────────────────────────────────────────────────────
# FEMNIST download: https://leaf.cmu.edu/
# After downloading, place the data in data/femnist/
# The LEAF partition script naturally partitions by writer identity.

class FEMNISTDataset(Dataset):
    """Loads FEMNIST from LEAF JSON files.
    Expected structure:
        data/femnist/train/all_data_iid_01_0_keep_5_train_9.json
        data/femnist/test/all_data_iid_01_0_keep_5_test_9.json
    """
    def __init__(self, data_dir: str, split: str = "train", transform=None):
        self.transform = transform
        data_path = os.path.join(data_dir, "femnist", split)
        self.x, self.y = [], []
        for fname in sorted(os.listdir(data_path)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(data_path, fname)) as f:
                raw = json.load(f)
            for user_data in raw["user_data"].values():
                self.x.extend(user_data["x"])
                self.y.extend(user_data["y"])
        self.x = np.array(self.x, dtype=np.float32).reshape(-1, 1, 28, 28)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = torch.tensor(self.x[idx])
        if self.transform:
            img = self.transform(img)
        return img, int(self.y[idx])


def load_femnist() -> Tuple[Dataset, Dataset]:
    femnist_dir = os.path.join(DATA_DIR, "femnist")
    if not os.path.exists(femnist_dir):
        raise FileNotFoundError(
            "FEMNIST data not found. Download LEAF from https://leaf.cmu.edu/ "
            "and run: cd leaf && ./preprocess.sh -s niid --sf 0.05 -k 35 -t sample "
            "then place train/ and test/ json folders in data/femnist/"
        )
    tf = T.Normalize((0.5,), (0.5,))
    train_ds = FEMNISTDataset(DATA_DIR, split="train", transform=tf)
    test_ds  = FEMNISTDataset(DATA_DIR, split="test",  transform=tf)
    return train_ds, test_ds

# ─────────────────────────────────────────────────────────────────────────────
# Sentiment140
# ─────────────────────────────────────────────────────────────────────────────
# Download URL: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
# Distributed by user account (Section 6.1)

class Sentiment140Dataset(Dataset):
    """Tokenised Sentiment140 dataset.
    Download: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
    Place training.1600000.processed.noemoticon.csv in data/sentiment140/
    """
    def __init__(self, texts, labels, vocab: dict, seq_len: int = 64):
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len
        self.data = self._tokenise(texts, vocab)

    def _tokenise(self, texts: List[str], vocab: dict) -> torch.Tensor:
        seqs = []
        for t in texts:
            ids = [vocab.get(w, 1) for w in t.lower().split()[:self.seq_len]]
            ids = ids + [0] * (self.seq_len - len(ids))   # pad
            seqs.append(ids)
        return torch.tensor(seqs, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def load_sentiment140(seq_len: int = 64, vocab_size: int = 50000):
    csv_path = os.path.join(DATA_DIR, "sentiment140",
                            "training.1600000.processed.noemoticon.csv")
    zip_path = os.path.join(DATA_DIR, "sentiment140", "trainingandtestdata.zip")

    if not os.path.exists(csv_path):
        url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        os.makedirs(os.path.join(DATA_DIR, "sentiment140"), exist_ok=True)
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(os.path.join(DATA_DIR, "sentiment140"))

    texts, labels = [], []
    with open(csv_path, encoding="latin-1") as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(1 if row[0] == "4" else 0)
            texts.append(row[5])

    # Build vocabulary from training data
    from collections import Counter
    word_freq = Counter(w for t in texts for w in t.lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in word_freq.most_common(vocab_size - 2):
        vocab[word] = len(vocab)

    split = int(0.7 * len(texts))
    train_ds = Sentiment140Dataset(texts[:split],  labels[:split],  vocab, seq_len)
    test_ds  = Sentiment140Dataset(texts[split:],  labels[split:],  vocab, seq_len)
    return train_ds, test_ds, vocab

# ─────────────────────────────────────────────────────────────────────────────
# Non-IID Dirichlet Partitioning  (Section 6.1, α_Dir = 0.5)
# ─────────────────────────────────────────────────────────────────────────────

def dirichlet_partition(
        dataset: Dataset,
        n_clients: int,
        alpha: float,
        seed: int = SEED,
) -> List[List[int]]:
    """Partition a dataset into n_clients subsets using Dirichlet(α) distribution.
    Section 6.1: 'Dirichlet distribution with concentration parameter α_Dir=0.5'
    Returns list of index lists, one per client.
    """
    rng = np.random.default_rng(seed)
    targets = np.array([y for _, y in dataset])
    n_classes = len(np.unique(targets))

    # Per-class indices
    class_indices = [np.where(targets == c)[0] for c in range(n_classes)]
    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for c_indices in class_indices:
        rng.shuffle(c_indices)
        proportions = rng.dirichlet([alpha] * n_clients)
        # Scale proportions to actual counts
        cumulative = (np.cumsum(proportions) * len(c_indices)).astype(int)
        cumulative[-1] = len(c_indices)
        start = 0
        for client_id, end in enumerate(cumulative):
            client_indices[client_id].extend(c_indices[start:end].tolist())
            start = end

    return client_indices


def writer_partition_femnist(dataset: FEMNISTDataset) -> List[List[int]]:
    """FEMNIST naturally partitions by writer identity (Section 6.1).
    This function simply passes through since LEAF already handles partitioning.
    """
    return [list(range(len(dataset)))]   # return full for single-node

# ─────────────────────────────────────────────────────────────────────────────
# Small root dataset for DT-RoA server trust computation  (Section 4.2)
# ─────────────────────────────────────────────────────────────────────────────

def get_root_dataset(dataset: Dataset, n_root: int = 100, seed: int = SEED) -> Dataset:
    """Sample n_root=100 clean examples for server root dataset (Section 4.2).
    'Droot = 100 samples drawn uniformly from the training distribution'
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_root, replace=False).tolist()
    return Subset(dataset, indices)

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
        dataset_name: str = DATASET,
        n_clients:    int  = N_CLIENTS,
        batch_size:   int  = BATCH_SIZE,
        alpha:        float = DIRICHLET_ALPHA,
) -> Tuple[List[DataLoader], DataLoader, DataLoader, Dataset]:
    """Return (client_loaders, val_loader, test_loader, root_dataset).
    client_loaders[i] is the DataLoader for client i's non-IID partition.
    """
    print(f"[dataset] Loading {dataset_name} with Dirichlet α={alpha}, N={n_clients}")

    if dataset_name in ("cifar10", "cifar100"):
        train_ds, test_ds = load_cifar(dataset_name)
    elif dataset_name == "femnist":
        train_ds, test_ds = load_femnist()
    elif dataset_name == "sentiment140":
        train_ds, test_ds, _ = load_sentiment140()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Train / val split (80/10/10 for CIFAR; 70/15/15 for Sentiment140)
    cfg = DATASET_CONFIG[dataset_name]
    tr, va, te = cfg["split"]
    n_total = len(train_ds)
    n_val   = int(n_total * va / (tr + va))
    n_tr    = n_total - n_val

    rng = torch.Generator().manual_seed(SEED)
    train_subset, val_subset = torch.utils.data.random_split(
        train_ds, [n_tr, n_val], generator=rng)

    # Dirichlet partition of train_subset into client shards
    # We re-index into the train_subset
    indices_in_trainset = train_subset.indices if hasattr(train_subset, "indices") else list(range(n_tr))
    tmp_ds = Subset(train_ds, indices_in_trainset)

    client_idx_lists = dirichlet_partition(tmp_ds, n_clients, alpha, seed=SEED)

    client_loaders = []
    for cid, idx_list in enumerate(client_idx_lists):
        subset = Subset(tmp_ds, idx_list)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=True)
        client_loaders.append(loader)

    val_loader  = DataLoader(val_subset, batch_size=batch_size * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds,   batch_size=batch_size * 2, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    root_ds = get_root_dataset(train_ds, n_root=100, seed=SEED)

    print(f"[dataset] Train shards: {n_clients}  "
          f"Val size: {len(val_subset)}  Test size: {len(test_ds)}")
    return client_loaders, val_loader, test_loader, root_ds
