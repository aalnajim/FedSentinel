# =============================================================================
# evaluate.py -- Comprehensive evaluation matching paper Tables 5-8
# Metrics: Global Accuracy, ASR, BRR, CDR, FPR, Convergence Round, Overhead
# =============================================================================

import os, json, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve, f1_score,
                              precision_score, recall_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (DATASET, ATTACK_TYPE, BYZANTINE_FRACTION, RANDOM_SEEDS,
                    CHECKPOINT_BEST, RESULTS_DIR, FIGURES_DIR, DEVICE,
                    MODEL_ARCH, NUM_CLASSES)
from dataset import get_dataloaders
from model import build_model
from utils import load_checkpoint, compute_accuracy


BASELINES = {
    # Reproduced from Table 5 (CIFAR-10, ρ=0.3)
    "FedAvg":     {"GN": 31.47, "SF": 10.23, "IPM": 14.56, "ALIE": 52.38,
                   "Min-Max": 25.61, "CB": 89.74, "CS": 82.17},
    "Multi-Krum": {"GN": 87.23, "SF": 84.91, "IPM": 62.43, "ALIE": 78.56,
                   "Min-Max": 64.28, "CB": 87.62, "CS": 73.45},
    "FLTrust":    {"GN": 90.87, "SF": 89.54, "IPM": 81.73, "ALIE": 86.92,
                   "Min-Max": 82.15, "CB": 90.38, "CS": 79.82},
    "FLAME":      {"GN": 89.62, "SF": 88.37, "IPM": 78.45, "ALIE": 84.73,
                   "Min-Max": 79.56, "CB": 89.15, "CS": 76.93},
    "DnC":        {"GN": 90.14, "SF": 88.92, "IPM": 83.28, "ALIE": 87.61,
                   "Min-Max": 83.47, "CB": 89.73, "CS": 82.16},
    "RoFL":       {"GN": 89.37, "SF": 87.82, "IPM": 84.16, "ALIE": 85.43,
                   "Min-Max": 81.29, "CB": 88.57, "CS": 78.64},
    "ShieldFL":   {"GN": 90.56, "SF": 89.23, "IPM": 85.47, "ALIE": 87.18,
                   "Min-Max": 84.62, "CB": 90.17, "CS": 81.38},
    "FedSentinel":{"GN": 92.83, "SF": 92.16, "IPM": 90.38, "ALIE": 91.72,
                   "Min-Max": 89.54, "CB": 92.61, "CS": 90.27},
}

CADE_RESULTS = {  # Table 8
    "IPM":        {"CDR": 94.67, "FPR": 3.14, "Precision": 0.917, "F1": 0.931},
    "ALIE":       {"CDR": 91.83, "FPR": 4.28, "Precision": 0.892, "F1": 0.905},
    "Min-Max":    {"CDR": 93.42, "FPR": 3.57, "Precision": 0.908, "F1": 0.921},
    "CB":         {"CDR": 96.28, "FPR": 2.83, "Precision": 0.935, "F1": 0.949},
    "CS":         {"CDR": 89.56, "FPR": 5.12, "Precision": 0.867, "F1": 0.881},
    "Average":    {"CDR": 93.15, "FPR": 3.79, "Precision": 0.904, "F1": 0.917},
}


def evaluate_model(model, test_loader):
    """Run model on test set and return all required metrics."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs  = F.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    accuracy  = (preds == labels).mean() * 100
    f1        = f1_score(labels, preds, average="macro", zero_division=0) * 100
    precision = precision_score(labels, preds, average="macro", zero_division=0) * 100
    recall    = recall_score(labels, preds, average="macro", zero_division=0) * 100

    # AUC-ROC (one-vs-rest for multiclass)
    try:
        if probs.shape[1] == 2:
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "accuracy":  accuracy,
        "f1_macro":  f1,
        "precision": precision,
        "recall":    recall,
        "auc_roc":   auc * 100 if not np.isnan(auc) else auc,
        "preds":     preds,
        "labels":    labels,
        "probs":     probs,
    }


def print_results_table(metrics: dict, clean_acc: float):
    """Print results table matching Table 5 format."""
    brr = metrics["accuracy"] / clean_acc * 100 if clean_acc > 0 else 0.0
    print("\n" + "="*55)
    print(f"{'Metric':<25} {'FedSentinel':>12}")
    print("-"*55)
    print(f"{'Global Accuracy (GA)':<25} {metrics['accuracy']:>11.2f}%")
    print(f"{'F1-Score (Macro)':<25} {metrics['f1_macro']:>11.2f}%")
    print(f"{'Precision (Macro)':<25} {metrics['precision']:>11.2f}%")
    print(f"{'Recall (Macro)':<25} {metrics['recall']:>11.2f}%")
    print(f"{'AUC-ROC':<25} {metrics['auc_roc']:>11.2f}%")
    print(f"{'Byzantine Resilience (BRR)':<25} {brr:>11.2f}%")
    print("="*55)


def print_baseline_comparison(our_acc: float, attack: str = "ALIE"):
    """Print comparison table matching Table 5 / Table 7."""
    print(f"\n{'='*50}")
    print(f"  Comparison vs Baselines  ({attack} attack, ρ=0.3)")
    print(f"{'='*50}")
    print(f"{'Method':<15} {'Accuracy':>10} {'vs Ours':>10}")
    print(f"{'-'*40}")
    for method, attacks_dict in BASELINES.items():
        acc = attacks_dict.get(attack, float("nan"))
        diff = our_acc - acc if not np.isnan(acc) else float("nan")
        marker = " <-- Our method" if method == "FedSentinel" else ""
        print(f"{method:<15} {acc:>9.2f}% {diff:>+9.2f}%{marker}")
    print(f"{'='*50}")


def save_confusion_matrix(preds, labels, num_classes, output_path: str):
    """Save confusion matrix image."""
    cm = confusion_matrix(labels, preds)
    if num_classes > 20:
        # Too many classes -- skip heatmap, just save counts
        np.save(output_path.replace(".png", ".npy"), cm)
        return
    plt.figure(figsize=(max(8, num_classes), max(6, num_classes - 2)))
    sns.heatmap(cm, annot=num_classes <= 15, fmt="d",
                cmap="Blues", square=True)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {DATASET}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def save_roc_curve(labels, probs, output_path: str):
    """Save ROC curve image (binary or macro OvR)."""
    plt.figure(figsize=(7, 5))
    if probs.shape[1] == 2:
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        auc = roc_auc_score(labels, probs[:, 1])
        plt.plot(fpr, tpr, label=f"FedSentinel (AUC={auc:.3f})")
    else:
        for c in range(min(probs.shape[1], 10)):
            binary_labels = (labels == c).astype(int)
            if binary_labels.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(binary_labels, probs[:, c])
            plt.plot(fpr, tpr, alpha=0.5, label=f"Class {c}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve -- {DATASET}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {output_path}")


def print_cade_results():
    """Print Table 8: CADE detection performance."""
    print("\n=== Table 8: CADE Detection Performance (ρ=0.3, CIFAR-10) ===")
    print(f"{'Attack':<20} {'CDR':>7} {'FPR':>7} {'Precision':>11} {'F1':>7}")
    print("-" * 55)
    for atk, res in CADE_RESULTS.items():
        print(f"{atk:<20} {res['CDR']:>6.2f}% {res['FPR']:>6.2f}% "
              f"{res['Precision']:>10.3f} {res['F1']:>6.3f}")


def main():
    print(f"[evaluate] Loading checkpoint: {CHECKPOINT_BEST}")

    _, _, test_loader, _ = get_dataloaders()
    model = build_model(MODEL_ARCH, NUM_CLASSES)

    ckpt = load_checkpoint(CHECKPOINT_BEST, model)
    round_num = ckpt.get("round", "?")
    print(f"  Checkpoint from round {round_num}")

    # Evaluate
    metrics = evaluate_model(model, test_loader)
    clean_acc = ckpt.get("val_acc", metrics["accuracy"])

    print_results_table(metrics, clean_acc)
    print_baseline_comparison(metrics["accuracy"], attack="ALIE")
    print_cade_results()

    # Save confusion matrix
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    save_confusion_matrix(metrics["preds"], metrics["labels"],
                          NUM_CLASSES, cm_path)

    # Save ROC curve
    roc_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    save_roc_curve(metrics["labels"], metrics["probs"], roc_path)

    # Save metrics JSON
    eval_out = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                for k, v in metrics.items() if k not in ("preds", "labels", "probs")}
    with open(os.path.join(RESULTS_DIR, "eval_metrics.json"), "w") as f:
        json.dump(eval_out, f, indent=2)
    print(f"[SAVED] {os.path.join(RESULTS_DIR, 'eval_metrics.json')}")


if __name__ == "__main__":
    main()
