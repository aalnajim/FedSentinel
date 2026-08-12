# FedSentinel: Byzantine-Resilient Federated Learning with Cryptographic Gradient Attestation

**Paper:** FedSentinel: A Byzantine-Resilient Federated Learning Framework with Cryptographic Gradient Attestation Against Coordinated Model Poisoning Attacks

**Author:** Abdullah Abdulkarim Alnajim

**Journal:** Electronics (MDPI) — Manuscript under review

**Section:** Computer Science & Engineering

**Special Issue:** The Future of Cryptography: Trends and Emerging Technologies

**Code:** This repository provides the implementation and experimental code for the FedSentinel framework.

---

## Abstract

FedSentinel is a Byzantine-resilient federated learning framework that integrates cryptographic gradient attestation with adaptive trust-weighted aggregation to defend against coordinated model poisoning attacks. It achieves 91.36% average accuracy under 30% Byzantine adversaries on CIFAR-10, outperforming state-of-the-art defenses by 4.42-8.74% while reducing attack success rates by 67.3%.

---

## Framework Overview

FedSentinel consists of three interconnected modules (Fig. 2 in the paper):

**CGAP -- Cryptographic Gradient Attestation Protocol** (Section 4.1, Algorithm 3)
Employs Pedersen commitment schemes and Bulletproof-style zero-knowledge range proofs to verify that client gradient updates satisfy predefined norm (||g||_2 <= B) and direction (cos(g, g_ref) >= -δ) constraints before aggregation.

**DT-RoA -- Dynamic Trust-Weighted Robust Aggregation** (Section 4.2)
Maintains per-client trust scores τ_i = α·τ_crypto + β·τ_stat + γ·τ_hist using an exponentially weighted moving average with anomaly-triggered decay. Trust scores modulate aggregation weights.

**CADE -- Coordinated Attack Detection Engine** (Section 4.3, Algorithm 2)
Identifies colluding Byzantine clients through spectral analysis of gradient covariance matrices. Projects gradients via Johnson-Lindenstrauss random projection and detects coalition signatures in the eigenvalue distribution.

---

## Datasets

| Dataset | Samples | Classes | Model | Split |
|---|---|---|---|---|
| CIFAR-10 | 60,000 | 10 | ResNet-18 | 80/10/10 |
| CIFAR-100 | 60,000 | 100 | ResNet-34 | 80/10/10 |
| FEMNIST | 805,263 | 62 | CNN-4 | 80/10/10 |
| Sentiment140 | 1,600,000 | 2 | LSTM-2L | 70/15/15 |

### Download Instructions

**CIFAR-10 / CIFAR-100** -- Downloaded automatically by torchvision on first run.

**FEMNIST** -- LEAF benchmark:
```bash
git clone https://github.com/TalwalkarLab/leaf.git
cd leaf/data/femnist
./preprocess.sh -s niid --sf 0.05 -k 35 -t sample
cp -r data/ ../../FedSentinel_Implementation/data/femnist/
```

**Sentiment140**:
```
URL: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
Place: data/sentiment140/training.1600000.processed.noemoticon.csv
```
Or run: `python data/download_data.py`

---

## Data Folder Structure

```
data/
|-- cifar-10-batches-py/          <- auto-downloaded by torchvision
|-- cifar-100-python/             <- auto-downloaded by torchvision
|-- femnist/
|     |-- train/                  <- LEAF JSON files
|     `-- test/
`-- sentiment140/
      `-- training.1600000.processed.noemoticon.csv
```

---

## Installation

```bash
git clone https://github.com/[author]/FedSentinel.git
cd FedSentinel_Implementation
pip install -r requirements.txt
```

**Python 3.10 required** (as in Table 4 of the paper).

---

## How to Train

```bash
# CIFAR-10 with ALIE attack, 30% Byzantine, 100 clients (default)
python train.py

# Custom attack / dataset
# Edit config.py:
#   DATASET = "cifar100"
#   ATTACK_TYPE = "coordinated_stealth"
#   BYZANTINE_FRACTION = 0.4
python train.py
```

Training runs 3 seeds (42, 123, 456) and reports mean ± std, matching Table 4.

---

## How to Evaluate

```bash
python evaluate.py
```

Outputs:
- Global Accuracy, F1, Precision, Recall, AUC-ROC
- Byzantine Resilience Ratio (BRR)
- Comparison table vs all baselines (Table 5 format)
- CADE detection performance (Table 8)
- `results/confusion_matrix.png`
- `results/roc_curve.png`

---

## How to Run Inference

```bash
python inference.py --image /path/to/image.png
```

---

## Results (Table 5, CIFAR-10, ρ=0.3)

| Method | No Attack | GN | SF | IPM | ALIE | Min-Max | CB | CS | Average |
|---|---|---|---|---|---|---|---|---|---|
| FedAvg | 93.82 | 31.47 | 10.23 | 14.56 | 52.38 | 25.61 | 89.74 | 82.17 | 43.74 |
| Multi-Krum | 92.15 | 87.23 | 84.91 | 62.43 | 78.56 | 64.28 | 87.62 | 73.45 | 76.93 |
| FLTrust | 93.41 | 90.87 | 89.54 | 81.73 | 86.92 | 82.15 | 90.38 | 79.82 | 85.92 |
| FLAME | 92.78 | 89.62 | 88.37 | 78.45 | 84.73 | 79.56 | 89.15 | 76.93 | 83.83 |
| DnC | 93.06 | 90.14 | 88.92 | 83.28 | 87.61 | 83.47 | 89.73 | 82.16 | 86.47 |
| RoFL | 92.53 | 89.37 | 87.82 | 84.16 | 85.43 | 81.29 | 88.57 | 78.64 | 84.47 |
| ShieldFL | 92.89 | 90.56 | 89.23 | 85.47 | 87.18 | 84.62 | 90.17 | 81.38 | 86.94 |
| **FedSentinel** | **93.47** | **92.83** | **92.16** | **90.38** | **91.72** | **89.54** | **92.61** | **90.27** | **91.36** |

---

## Key Hyperparameters (Table 4)

| Parameter | Value |
|---|---|
| Clients N | 100 |
| Rounds T | 500 |
| Local Epochs K | 5 |
| Batch Size | 64 |
| Global LR η | 0.01 (cosine decay) |
| Trust (α, β, γ) | 0.3, 0.4, 0.3 |
| EWMA λ | 0.9 |
| Decay μ | 0.5 |
| CADE μ_CADE | 0.3 |
| Projection dim k | 64 |
| Chunk size s | 256 |
| Norm bound B | 10.0 |
| Direction δ | 0.5 |
| Clip threshold C | 10.0 |

---

## Repository Structure

```
FedSentinel_Implementation/
|
|-- figures/                <- 11 figures extracted from paper (HD PNG)
|-- data/                   <- Dataset storage + download_data.py
|-- checkpoints/            <- Saved model checkpoints
|-- results/                <- Evaluation outputs (metrics, plots)
|-- notebooks/
|     `-- demo.ipynb        <- Step-by-step walkthrough
|
|-- config.py               <- All hyperparameters (Table 4)
|-- dataset.py              <- Dataset loading + Dirichlet partitioning
|-- model.py                <- ResNet-18/34, CNN-4, LSTM-2L + attacks
|-- fedsentinel.py          <- CGAP, CADE, DT-RoA, FedSentinelServer
|-- train.py                <- Full federated training loop
|-- evaluate.py             <- Metrics + Tables 5-8
|-- inference.py            <- Single-sample inference
|-- utils.py                <- Helpers, visualization, metrics
|-- requirements.txt        <- Package versions
`-- README.md
```

---

## Citation

```bibtex
@unpublished{fedsentinel2026,
  author  = {Abdullah Abdulkarim Alnajim},
  title   = {FedSentinel: A Byzantine-Resilient Federated Learning Framework
             with Cryptographic Gradient Attestation Against Coordinated
             Model Poisoning Attacks},
  note    = {Manuscript submitted to Electronics, Section: Computer Science \& Engineering,
             Special Issue: The Future of Cryptography: Trends and Emerging Technologies},
  year    = {2026}
}
```
