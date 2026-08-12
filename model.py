# =============================================================================
# model.py -- Model architectures used in FedSentinel experiments
# Paper Table 3:
#   CIFAR-10    -> ResNet-18
#   CIFAR-100   -> ResNet-34
#   FEMNIST     -> CNN-4 (4-layer CNN)
#   Sentiment140 -> LSTM-2L (2-layer LSTM)
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from typing import Optional
from config import (DATASET, NUM_CLASSES, MODEL_ARCH, DEVICE,
                    DATASET_CONFIG)


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-18 for CIFAR-10  (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet18(num_classes: int = 10) -> nn.Module:
    """ResNet-18 adapted for CIFAR-32x32 images.
    Replaces the original 7x7 stride-2 conv + maxpool with a 3x3 stride-1 conv.
    Standard practice for small-image ResNets.
    """
    model = tvm.resnet18(weights=None)
    # Adapt for 32x32 CIFAR images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# ResNet-34 for CIFAR-100  (Table 3)
# ─────────────────────────────────────────────────────────────────────────────

def build_resnet34(num_classes: int = 100) -> nn.Module:
    """ResNet-34 adapted for CIFAR-32x32 images."""
    model = tvm.resnet34(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CNN-4 for FEMNIST  (Table 3)
# 4-layer convolutional network for 28x28 grayscale handwritten characters
# ─────────────────────────────────────────────────────────────────────────────

class CNN4(nn.Module):
    """4-layer CNN for FEMNIST (62 character classes, 28x28 grayscale).
    Architecture follows standard LEAF benchmark CNN-4:
      Conv(1->32, 5x5) -> ReLU -> MaxPool(2)
      Conv(32->64, 5x5) -> ReLU -> MaxPool(2)
      FC(1024) -> ReLU -> Dropout(0.5) -> FC(num_classes)
    """
    def __init__(self, num_classes: int = 62):
        super().__init__()
        # Section 6 -- CNN-4 for FEMNIST
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),   # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28 -> 14x14
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block(x))


# ─────────────────────────────────────────────────────────────────────────────
# LSTM-2L for Sentiment140  (Table 3)
# 2-layer bidirectional LSTM for binary sentiment classification
# ─────────────────────────────────────────────────────────────────────────────

class LSTM2L(nn.Module):
    """2-layer LSTM for Sentiment140 binary classification.
    Architecture:
      Embedding(vocab_size, embed_dim=128)
      -> LSTM(128, hidden=256, num_layers=2, dropout=0.3)
      -> Linear(256, 2)
    """
    def __init__(self,
                 vocab_size: int  = 50000,
                 embed_dim:  int  = 128,
                 hidden_dim: int  = 256,
                 num_layers: int  = 2,
                 num_classes: int = 2,
                 dropout:    float = 0.3):
        super().__init__()
        cfg = DATASET_CONFIG.get("sentiment140", {})
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout  = nn.Dropout(dropout)
        self.fc       = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token ids
        emb = self.dropout(self.embedding(x))          # (B, L, E)
        out, (h_n, _) = self.lstm(emb)                 # h_n: (layers, B, H)
        last_hidden = self.dropout(h_n[-1])             # (B, H)
        return self.fc(last_hidden)                     # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Model factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(arch: str = MODEL_ARCH, num_classes: int = NUM_CLASSES,
                **kwargs) -> nn.Module:
    """Return the model specified by arch string."""
    arch = arch.lower()
    if arch == "resnet18":
        model = build_resnet18(num_classes)
    elif arch == "resnet34":
        model = build_resnet34(num_classes)
    elif arch == "cnn4":
        model = CNN4(num_classes)
    elif arch == "lstm2l":
        model = LSTM2L(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model.to(DEVICE)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector (the 'gradient' in FL)."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params(model: nn.Module, flat: torch.Tensor) -> None:
    """Set model parameters from a flat vector."""
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(p.shape))
        offset += n


# ─────────────────────────────────────────────────────────────────────────────
# Attack implementations  (Section 6.3)
# ─────────────────────────────────────────────────────────────────────────────

class AttackFactory:
    """Implements all seven attack strategies from Section 6.3."""

    @staticmethod
    def gaussian_noise(grad: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        """GN: Add Gaussian noise N(0, σ²I) -- Section 6.3."""
        return grad + sigma * torch.randn_like(grad)

    @staticmethod
    def sign_flipping(grad: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """SF: g_i = -κ * g_i -- Section 6.3."""
        return -kappa * grad

    @staticmethod
    def inner_product_manipulation(
            grad: torch.Tensor,
            honest_grads: torch.Tensor,
    ) -> torch.Tensor:
        """IPM [28]: maximize inner product with negative true gradient.
        Crafts update: g_mal = -c * mean(honest_grads)
        """
        mean_honest = honest_grads.mean(dim=0)
        # Scale to same norm as mean honest gradient
        scale = grad.norm() / (mean_honest.norm() + 1e-8)
        return -scale * mean_honest

    @staticmethod
    def alie(
            grad: torch.Tensor,
            honest_grads: torch.Tensor,
            z: float = 2.0,
    ) -> torch.Tensor:
        """ALIE [22]: A Little is Enough attack.
        Submits updates at boundary of honest distribution to evade detection.
        g_mal = mean + z * std
        """
        mu  = honest_grads.mean(dim=0)
        std = honest_grads.std(dim=0)
        return mu + z * std

    @staticmethod
    def min_max(
            grad: torch.Tensor,
            honest_grads: torch.Tensor,
            gamma: float = 2.0,
    ) -> torch.Tensor:
        """Min-Max [13]: maximize deviation while minimizing distance to nearest honest update.
        Approximation: reflect and scale beyond nearest honest update.
        """
        mean_honest = honest_grads.mean(dim=0)
        deviation   = grad - mean_honest
        return mean_honest - gamma * deviation

    @staticmethod
    def coordinated_backdoor(
            grad: torch.Tensor,
            trigger_mask: torch.Tensor,
            target_class: int,
            scale: float = 10.0,
    ) -> torch.Tensor:
        """CB: Embed pixel-pattern backdoor trigger.
        Byzantine clients amplify gradient component aligning with backdoor direction.
        trigger_mask: binary mask of trigger pixels in flattened parameter space.
        """
        poisoned = grad.clone()
        poisoned[trigger_mask] *= scale
        return poisoned

    @staticmethod
    def coordinated_stealth(
            grad: torch.Tensor,
            round_idx: int,
            total_rounds: int,
            n_byzantine: int,
            budget: float = 1.0,
    ) -> torch.Tensor:
        """CS: Coordinated Stealth -- partition poisoning across rounds.
        Each Byzantine client contributes a fraction (budget / total_rounds) of
        poisoning per round to maintain per-round statistical consistency.
        """
        per_round_scale = budget / max(1, total_rounds)
        # Accumulate small gradient perturbation aligned with negative direction
        poison = -per_round_scale * grad
        return grad + poison


def apply_attack(
        gradients: torch.Tensor,     # (N_byzantine, d)
        honest_gradients: torch.Tensor,  # (N_honest, d)
        attack_type: str,
        round_idx: int = 0,
        total_rounds: int = 500,
        **kwargs,
) -> torch.Tensor:
    """Apply the specified attack to Byzantine client gradients."""
    results = []
    for i in range(gradients.shape[0]):
        g = gradients[i]
        if attack_type == "gaussian_noise":
            g = AttackFactory.gaussian_noise(g, kwargs.get("sigma", 0.1))
        elif attack_type == "sign_flipping":
            g = AttackFactory.sign_flipping(g, kwargs.get("kappa", 1.0))
        elif attack_type == "ipm":
            g = AttackFactory.inner_product_manipulation(g, honest_gradients)
        elif attack_type == "alie":
            g = AttackFactory.alie(g, honest_gradients)
        elif attack_type == "min_max":
            g = AttackFactory.min_max(g, honest_gradients)
        elif attack_type == "coordinated_backdoor":
            mask = torch.zeros_like(g, dtype=torch.bool)
            mask[:100] = True  # simple trigger mask for demonstration
            g = AttackFactory.coordinated_backdoor(g, mask, kwargs.get("target", 0))
        elif attack_type == "coordinated_stealth":
            g = AttackFactory.coordinated_stealth(g, round_idx, total_rounds,
                                                   gradients.shape[0])
        elif attack_type == "no_attack":
            pass  # benign
        results.append(g)
    return torch.stack(results)


if __name__ == "__main__":
    m = build_model("resnet18", 10)
    print(f"ResNet-18 parameters: {count_parameters(m):,}")
    m34 = build_model("resnet34", 100)
    print(f"ResNet-34 parameters: {count_parameters(m34):,}")
    cnn = build_model("cnn4", 62)
    print(f"CNN-4 parameters: {count_parameters(cnn):,}")
    lstm = build_model("lstm2l", 2)
    print(f"LSTM-2L parameters: {count_parameters(lstm):,}")
