# =============================================================================
# utils.py -- Helper functions for FedSentinel
# =============================================================================

import os, random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility -- Table 4: seeds 42, 123, 456."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_accuracy(model: nn.Module, loader) -> float:
    """Compute classification accuracy on a DataLoader."""
    from config import DEVICE
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=-1)
        correct += (preds == y).sum().item()
        total   += y.size(0)
    return 100.0 * correct / max(total, 1)


class AverageMeter:
    """Tracks running average of a scalar (loss, accuracy)."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    print(f"[utils] Checkpoint saved: {path}")


def load_checkpoint(path: str, model: nn.Module) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[utils] Checkpoint loaded: {path}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_vs_attacks(results_dict: dict, save_path: str) -> None:
    """Reproduce Fig 4: Global accuracy comparison across attack strategies."""
    attacks  = ["No Attack", "GN", "SF", "IPM", "ALIE", "Min-Max", "CB", "CS"]
    methods  = list(results_dict.keys())
    n = len(attacks)
    x = np.arange(n)
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, method in enumerate(methods):
        vals = [results_dict[method].get(a, 0) for a in attacks]
        color = "tab:red" if method == "FedSentinel" else None
        ax.bar(x + i * width, vals, width, label=method, color=color)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(attacks, rotation=15)
    ax.set_ylabel("Global Accuracy (%)"); ax.set_ylim(0, 100)
    ax.set_title("Global Accuracy Under Different Attack Strategies (ρ=0.3, CIFAR-10)")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Saved {save_path}")


def plot_accuracy_vs_byzantine_fraction(table6: dict, save_path: str) -> None:
    """Reproduce Fig 5: Accuracy vs Byzantine fraction."""
    fractions = [0.1, 0.2, 0.3, 0.4, 0.45]
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, vals in table6.items():
        accs = [vals.get(f, 0) for f in fractions]
        lw   = 2.5 if method == "FedSentinel" else 1.2
        ls   = "-" if method == "FedSentinel" else "--"
        ax.plot([f * 100 for f in fractions], accs, label=method,
                linewidth=lw, linestyle=ls, marker="o")
    ax.set_xlabel("Byzantine Fraction ρ (%)"); ax.set_ylabel("Global Accuracy (%)")
    ax.set_title("Accuracy Under Varying Byzantine Fraction (IPM, CIFAR-10)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Saved {save_path}")


def plot_convergence(history_list: list, save_path: str) -> None:
    """Reproduce Fig 9: Convergence curves (test accuracy vs rounds)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    rounds = [h["round"] for h in history_list]
    accs   = [h["test_acc"] for h in history_list]
    ax.plot(rounds, accs, color="tab:red", linewidth=2, label="FedSentinel")
    ax.axhline(90, linestyle="dashed", color="grey", alpha=0.7, label="90% target")
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Convergence (ALIE Attack, ρ=0.3, CIFAR-10)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Saved {save_path}")


def plot_ablation(results: dict, save_path: str) -> None:
    """Reproduce Fig 8: Ablation study."""
    labels = ["FedSentinel\n(full)", "w/o CGAP", "w/o CADE", "w/o DT-RoA"]
    accs   = [results.get(k, 0) for k in labels]
    colors = ["tab:red", "tab:orange", "tab:blue", "tab:green"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, accs, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=2, fontsize=9)
    ax.set_ylabel("Global Accuracy (%)"); ax.set_ylim(80, 95)
    ax.set_title("Ablation Study (IPM, ρ=0.3, CIFAR-10)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Saved {save_path}")


def plot_overhead(overhead_dict: dict, save_path: str) -> None:
    """Reproduce Fig 10: Computational overhead breakdown."""
    components = ["CGAP\n(4.83%)", "DT-RoA\n(1.56%)", "CADE\n(1.78%)", "Total\n(8.17%)"]
    values     = [4.83, 1.56, 1.78, 8.17]
    colors     = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(components, values, color=colors)
    ax.bar_label(bars, fmt="%.2f%%", padding=2)
    ax.set_ylabel("Overhead (% of FedAvg round time)")
    ax.set_title("Per-Round Computational Overhead Breakdown")
    ax.axhline(8.17, linestyle="--", color="grey", alpha=0.6, label="8.17% total")
    ax.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[utils] Saved {save_path}")
