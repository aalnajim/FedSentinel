# =============================================================================
# config.py -- Centralised configuration for FedSentinel
# Paper: "FedSentinel: A Byzantine-Resilient Federated Learning Framework
#         with Cryptographic Gradient Attestation Against Coordinated
#         Model Poisoning Attacks"
# All hyperparameters from Table 4 and Section 6 of the paper
# =============================================================================

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
FIGURES_DIR     = os.path.join(BASE_DIR, "figures")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

for d in [DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Dataset settings (Table 3) ─────────────────────────────────────────────
DATASET_CONFIG = {
    "cifar10": {
        "name":       "CIFAR-10",
        "num_classes": 10,
        "model":      "resnet18",
        "split":      (0.8, 0.1, 0.1),        # train/val/test
        "input_size": (3, 32, 32),
        "url":        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    },
    "cifar100": {
        "name":       "CIFAR-100",
        "num_classes": 100,
        "model":      "resnet34",
        "split":      (0.8, 0.1, 0.1),
        "input_size": (3, 32, 32),
        "url":        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    },
    "femnist": {
        "name":       "FEMNIST",
        "num_classes": 62,
        "model":      "cnn4",
        "split":      (0.8, 0.1, 0.1),
        "input_size": (1, 28, 28),
        "url":        "https://leaf.cmu.edu/",          # LEAF benchmark
    },
    "sentiment140": {
        "name":       "Sentiment140",
        "num_classes": 2,
        "model":      "lstm2l",
        "split":      (0.7, 0.15, 0.15),
        "url":        "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip",
        "vocab_size": 50000,
        "seq_len":    64,
        "embed_dim":  128,
    },
}

# Active dataset for this run
DATASET = "cifar10"
NUM_CLASSES = DATASET_CONFIG[DATASET]["num_classes"]
MODEL_ARCH  = DATASET_CONFIG[DATASET]["model"]

# ── Federated Learning settings (Table 4) ─────────────────────────────────
N_CLIENTS       = 100          # Total participating clients
N_ROUNDS        = 500          # Communication rounds T
K_LOCAL_EPOCHS  = 5            # Local epochs K per round
BATCH_SIZE      = 64           # Batch size per client (Table 4)
RANDOM_SEEDS    = [42, 123, 456]   # 3-run average (Table 4)

# Data heterogeneity: Dirichlet concentration for non-IID split
DIRICHLET_ALPHA = 0.5          # alpha_Dir = 0.5 (Section 6.1)

# ── Optimizer settings (Table 4) ──────────────────────────────────────────
OPTIMIZER       = "sgd"
MOMENTUM        = 0.9
GLOBAL_LR       = 0.01         # η -- cosine decay (Table 4)
LOCAL_LR        = 0.01         # local learning rate (Table 4)
LR_SCHEDULER    = "cosine"     # cosine decay

# ── Threat model settings (Section 3.2) ───────────────────────────────────
BYZANTINE_FRACTION = 0.3       # ρ = 0.3 (main experiment)
BYZANTINE_FRACTIONS_EVAL = [0.1, 0.2, 0.3, 0.4, 0.45]  # Table 6

# Attack types
ATTACKS = [
    "no_attack",
    "gaussian_noise",    # GN: N(0, σ²I)
    "sign_flipping",     # SF: g_i = -κ * g_i
    "ipm",               # Inner Product Manipulation
    "alie",              # A Little is Enough
    "min_max",           # Min-Max optimization attack
    "coordinated_backdoor",   # CB: pixel-pattern backdoor
    "coordinated_stealth",    # CS: cross-round partitioned poisoning
]
ATTACK_TYPE = "alie"           # default attack for training run

# Attack hyperparameters
GN_SIGMA        = 0.1          # Gaussian noise std
SF_KAPPA        = 1.0          # Sign-flip scaling factor
BACKDOOR_TARGET = 0            # Target class for backdoor

# ── CGAP settings (Section 4.1) ───────────────────────────────────────────
CGAP_NORM_BOUND     = 10.0     # B -- gradient norm bound
CGAP_DIRECTION_DELTA = 0.5     # δ -- max anti-alignment
CGAP_CHUNK_SIZE     = 256      # s -- commitment chunk size
CGAP_QUANTIZE_BITS  = 8        # b -- quantization bit-width
CGAP_ENABLED        = True

# ── DT-RoA trust settings (Section 4.2, Table 4) ─────────────────────────
TRUST_ALPHA     = 0.3          # α -- crypto trust weight
TRUST_BETA      = 0.4          # β -- statistical trust weight
TRUST_GAMMA     = 0.3          # γ -- historical trust weight
# α + β + γ = 1.0 ✓

EWMA_LAMBDA     = 0.9          # λ -- EWMA smoothing factor
DECAY_MU        = 0.5          # μ -- attestation failure decay
DECAY_MU_CADE   = 0.3          # μ_CADE -- coalition detection decay
ROOT_DATASET_SIZE = 100        # |D_root| server clean samples
CLIP_THRESHOLD  = 10.0         # C -- gradient clipping

# ── CADE settings (Section 4.3) ───────────────────────────────────────────
CADE_PROJ_DIM       = 64       # k -- random projection dimension
CADE_WARMUP_ROUNDS  = 20       # T_warmup -- calibration rounds
CADE_SPECTRAL_PCTL  = 95       # θ_s percentile (95th)
CADE_PROJ_THRESHOLD = 2.5      # θ_p -- projection score multiplier
CADE_ENABLED        = True

# ── Model checkpoint paths ─────────────────────────────────────────────────
CHECKPOINT_BEST   = os.path.join(CHECKPOINT_DIR, "fedsentinel_best.pt")
CHECKPOINT_LAST   = os.path.join(CHECKPOINT_DIR, "fedsentinel_last.pt")
RESULTS_CSV       = os.path.join(RESULTS_DIR,    "results.csv")
TENSORBOARD_DIR   = os.path.join(LOGS_DIR,       "tensorboard")

# ── Reproducibility ───────────────────────────────────────────────────────
SEED = 42

# ── Hardware ──────────────────────────────────────────────────────────────
DEVICE = "cuda"    # "cuda" or "cpu"
NUM_WORKERS = 4

# ── Logging / evaluation frequency ───────────────────────────────────────
EVAL_EVERY    = 10    # evaluate global model every N rounds
LOG_EVERY     = 1     # log metrics every N rounds
SAVE_EVERY    = 50    # checkpoint every N rounds

print("[config.py] FedSentinel configuration loaded.")
print(f"  Dataset:    {DATASET}  ({NUM_CLASSES} classes)")
print(f"  Clients:    {N_CLIENTS}  |  Rounds: {N_ROUNDS}")
print(f"  Attack:     {ATTACK_TYPE}  |  Byzantine fraction: {BYZANTINE_FRACTION}")
print(f"  Trust (α,β,γ) = ({TRUST_ALPHA},{TRUST_BETA},{TRUST_GAMMA})")
