# =============================================================================
# fedsentinel.py -- Core FedSentinel framework implementation
# Paper Section 4: CGAP (Alg 3), CADE (Alg 2), DT-RoA, FedSentinel (Alg 1)
# Equations: (8)-(23)
# =============================================================================

import math
import copy
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_CLIENTS, K_LOCAL_EPOCHS, LOCAL_LR, BATCH_SIZE,
    CGAP_NORM_BOUND, CGAP_DIRECTION_DELTA, CGAP_CHUNK_SIZE, CGAP_QUANTIZE_BITS,
    CGAP_ENABLED,
    TRUST_ALPHA, TRUST_BETA, TRUST_GAMMA, EWMA_LAMBDA, DECAY_MU, DECAY_MU_CADE,
    CLIP_THRESHOLD,
    CADE_PROJ_DIM, CADE_WARMUP_ROUNDS, CADE_SPECTRAL_PCTL, CADE_PROJ_THRESHOLD,
    CADE_ENABLED, DEVICE,
)
from model import get_flat_params, set_flat_params


# =============================================================================
# CGAP: Cryptographic Gradient Attestation Protocol  (Section 4.1, Algorithm 3)
# NOTE: Full ZK-proofs require dedicated crypto libraries (PyCryptodome 3.19 +
#       Bulletproofs). Here we implement the constraint-checking and quantization
#       logic. The commitment and proof generation are simulated for the FL
#       training loop; replace with PyCryptodome calls for production.
# =============================================================================

class CGAPModule:
    """Cryptographic Gradient Attestation Protocol.
    Algorithm 3 from the paper.
    Equations (8)-(13).
    """

    def __init__(self,
                 norm_bound: float = CGAP_NORM_BOUND,
                 delta: float      = CGAP_DIRECTION_DELTA,
                 chunk_size: int   = CGAP_CHUNK_SIZE,
                 bits: int         = CGAP_QUANTIZE_BITS):
        self.B       = norm_bound    # Eq (7): ||g||_2 <= B
        self.delta   = delta         # Eq (12): direction threshold
        self.s       = chunk_size    # Section 4.1: s=256
        self.bits    = bits          # b=8 bit quantization
        self.Delta   = norm_bound / (2 ** (bits - 1) - 1)  # Eq (9)

    def quantize(self, grad: torch.Tensor) -> torch.Tensor:
        """Stochastic quantization -- Equation (9).
        g~_{i,j} = sign(g_{i,j}) * floor(|g_{i,j}| / Delta + xi)
        xi ~ Uniform[0, 1)
        """
        xi    = torch.rand_like(grad)
        q     = torch.sign(grad) * torch.floor(grad.abs() / self.Delta + xi)
        q     = q.clamp(-2**(self.bits-1), 2**(self.bits-1) - 1)
        return q

    def dequantize(self, q: torch.Tensor) -> torch.Tensor:
        return q * self.Delta

    def commit(self, grad_q: torch.Tensor) -> bytes:
        """Simulated Pedersen commitment -- Equation (8).
        In production: use PyCryptodome elliptic-curve group operations.
        Returns a SHA-256 hash of the gradient as a commitment placeholder.
        """
        import hashlib
        data = grad_q.cpu().numpy().tobytes()
        return hashlib.sha256(data).digest()

    def verify_norm(self, grad_q: torch.Tensor) -> bool:
        """Verify ||g~||_2 <= B -- Equation (7) / (11).
        In production: verify Bulletproof range proof.
        """
        g_float = self.dequantize(grad_q)
        return float(g_float.norm(p=2)) <= self.B * (1 + 1e-4)   # tiny tolerance

    def verify_direction(self,
                         grad_q: torch.Tensor,
                         ref_grad: Optional[torch.Tensor]) -> bool:
        """Verify direction constraint <g, g_ref> >= -delta*||g||*||g_ref|| -- Eq (12).
        Skipped at round t=0 (no reference gradient).
        """
        if ref_grad is None:
            return True   # Section 4.4: omitted at t=0
        g = self.dequantize(grad_q)
        cos_sim = F.cosine_similarity(g.unsqueeze(0), ref_grad.unsqueeze(0)).item()
        return cos_sim >= -self.delta

    def attest(self,
               grad: torch.Tensor,
               ref_grad: Optional[torch.Tensor] = None) -> Tuple[bool, torch.Tensor, bytes]:
        """Full CGAP attestation -- Algorithm 3 (Equation 13).
        Returns:
            v_i: bool -- pass/fail verification
            grad_q: quantized gradient (transmitted to server)
            commitment: commitment bytes
        """
        # Section 4.1 Algorithm 3 Steps 1-11
        grad_q     = self.quantize(grad)               # Step 1: quantize
        commitment = self.commit(grad_q)               # Steps 4-7: commit
        pass_norm  = self.verify_norm(grad_q)          # Step 9: norm proof
        pass_dir   = self.verify_direction(grad_q, ref_grad)  # Step 10: dir proof
        v_i = pass_norm and pass_dir                   # Eq (13)
        return v_i, grad_q, commitment

    def batch_verify(self,
                     grads_q: List[torch.Tensor],
                     ref_grad: Optional[torch.Tensor]) -> List[bool]:
        """Server-side batch verification -- Section 4.1 batch verification.
        'Reducing verification complexity from O(N*d) to O(N+d) group exponentiations.'
        """
        return [
            self.verify_norm(g) and self.verify_direction(g, ref_grad)
            for g in grads_q
        ]


# =============================================================================
# CADE: Coordinated Attack Detection Engine  (Section 4.3, Algorithm 2)
# Equations (21)-(23)
# =============================================================================

class CADEModule:
    """Coordinated Attack Detection Engine.
    Algorithm 2 from the paper.
    Equations (21)-(23).
    """

    def __init__(self,
                 proj_dim:      int   = CADE_PROJ_DIM,
                 warmup_rounds: int   = CADE_WARMUP_ROUNDS,
                 spectral_pctl: float = CADE_SPECTRAL_PCTL,
                 proj_threshold: float = CADE_PROJ_THRESHOLD):
        self.k         = proj_dim       # projection dimension
        self.T_warmup  = warmup_rounds  # warm-up calibration rounds
        self.pctl      = spectral_pctl  # 95th percentile threshold
        self.theta_p   = proj_threshold # θ_p = 2.5
        self.theta_s   = None           # calibrated adaptively
        self.warmup_scores: List[float] = []
        self.projection: Optional[torch.Tensor] = None

    def _get_projection(self, d: int) -> torch.Tensor:
        """Random Gaussian projection matrix R ∈ R^{d×k}.
        Johnson-Lindenstrauss: preserves pairwise distances.
        Algorithm 2 Step 1.
        """
        if self.projection is None or self.projection.shape[0] != d:
            self.projection = torch.randn(d, self.k, device=DEVICE) / math.sqrt(self.k)
        return self.projection

    def compute_spectral_score(self, G: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Compute spectral anomaly score -- Equation (22).
        Score_spectral = λ_1 / (Σ λ_j) - 1/k
        G: (N, d) gradient matrix of verified clients.
        """
        N, d = G.shape
        R = self._get_projection(d)

        # Algorithm 2 Steps 1-2: random projection
        G_proj = G @ R                                  # (N, k)

        # Algorithm 2 Step 3: centered covariance -- Equation (21)
        mean_g = G_proj.mean(dim=0, keepdim=True)
        G_c    = G_proj - mean_g                        # (N, k)
        Sigma  = (G_c.T @ G_c) / max(N - 1, 1)        # (k, k)

        # Algorithm 2 Step 4: eigendecomposition
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        except Exception:
            eigenvalues = torch.ones(self.k, device=DEVICE)
            eigenvectors = torch.eye(self.k, device=DEVICE)

        eigenvalues = eigenvalues.flip(0)   # descending order
        eigenvectors = eigenvectors.flip(1)

        # Algorithm 2 Step 5: spectral score -- Equation (22)
        lambda_sum = eigenvalues.sum().item()
        lambda_max = eigenvalues[0].item()
        score = (lambda_max / (lambda_sum + 1e-10)) - (1.0 / self.k)

        return score, eigenvectors[:, 0]   # score, principal eigenvector u_1

    def calibrate(self, score: float) -> None:
        """Record warmup spectral scores for adaptive threshold calibration."""
        self.warmup_scores.append(score)
        if len(self.warmup_scores) >= self.T_warmup:
            self.theta_s = float(np.percentile(self.warmup_scores, self.pctl))
            print(f"[CADE] Threshold calibrated: θ_s = {self.theta_s:.4f} "
                  f"(95th percentile over {self.T_warmup} warmup rounds)")

    def detect(self,
               verified_grads: torch.Tensor,    # (N_verified, d)
               client_ids:     List[int],
               round_idx:      int,
    ) -> List[int]:
        """Identify coalition members -- Algorithm 2.
        Returns list of flagged client IDs.
        """
        if verified_grads.shape[0] < 3:
            return []

        score, u1 = self.compute_spectral_score(verified_grads)

        # Warmup calibration phase
        if round_idx < self.T_warmup or self.theta_s is None:
            self.calibrate(score)
            return []   # no flagging during warmup

        # Algorithm 2 Step 6: check threshold
        if score <= self.theta_s:
            return []   # no coordination detected

        # Algorithm 2 Steps 7-11: coalition identification -- Equation (23)
        G_proj     = verified_grads @ self._get_projection(verified_grads.shape[1])
        mean_proj  = G_proj.mean(dim=0)
        G_proj_c   = G_proj - mean_proj

        # Project onto principal eigenvector (aligned to CADE's projection space)
        scores_i = (G_proj_c @ u1.unsqueeze(-1)).squeeze(-1)   # (N_verified,)

        median_score = scores_i.median().item()
        threshold    = self.theta_p * abs(median_score)

        flagged = [
            client_ids[i]
            for i, s in enumerate(scores_i.tolist())
            if abs(s) > threshold and s > 0   # high positive projection
        ]
        if flagged:
            print(f"[CADE] Round {round_idx}: Score={score:.4f} > θ_s={self.theta_s:.4f}. "
                  f"Flagged {len(flagged)} clients: {flagged}")
        return flagged


# =============================================================================
# DT-RoA: Dynamic Trust-Weighted Robust Aggregation  (Section 4.2)
# Equations (14)-(20)
# =============================================================================

class DTRoAModule:
    """Dynamic Trust-Weighted Robust Aggregation.
    Equations (14)-(20) from the paper.
    """

    def __init__(self,
                 n_clients:   int   = N_CLIENTS,
                 alpha:       float = TRUST_ALPHA,
                 beta:        float = TRUST_BETA,
                 gamma:       float = TRUST_GAMMA,
                 lam:         float = EWMA_LAMBDA,
                 mu:          float = DECAY_MU,
                 mu_cade:     float = DECAY_MU_CADE,
                 clip_C:      float = CLIP_THRESHOLD):
        self.N   = n_clients
        self.a   = alpha      # weight for τ_crypto
        self.b   = beta       # weight for τ_stat
        self.g   = gamma      # weight for τ_hist
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "α+β+γ must equal 1"
        self.lam     = lam
        self.mu      = mu
        self.mu_cade = mu_cade
        self.C       = clip_C
        # Initialize trust scores: τ_i^0 = 1/N (Algorithm 1 Step 1)
        self.tau_hist = torch.ones(n_clients) / n_clients

    def _clip(self, g: torch.Tensor) -> torch.Tensor:
        """Gradient clipping -- clip(g, C) in Equation (18)."""
        norm = g.norm(p=2)
        if norm > self.C:
            g = g * self.C / norm
        return g

    def compute_trust(self,
                      v_flags:       torch.Tensor,   # (N,) binary CGAP outcomes
                      stat_cosines:  torch.Tensor,   # (N,) cosine similarity with ref
                      cade_flagged:  List[int],
    ) -> torch.Tensor:
        """Compute per-client trust scores -- Equations (14)-(17).
        v_flags:      τ_crypto -- Eq (15)
        stat_cosines: τ_stat   -- Eq (16)
        EWMA update:  τ_hist   -- Eq (17)
        Combined:     τ         -- Eq (14)
        """
        tau_crypto = v_flags.float()                     # Eq (15)
        tau_stat   = stat_cosines.clamp(min=0.0)         # Eq (16): max(0, cosine)

        # Eq (17): EWMA update with anomaly-triggered decay
        for i in range(self.N):
            if i in cade_flagged:
                self.tau_hist[i] = self.mu_cade * self.tau_hist[i]  # CADE penalty (μ_CADE=0.3)
            elif v_flags[i].item() == 1:
                self.tau_hist[i] = (self.lam * self.tau_hist[i] +
                                    (1 - self.lam) * tau_stat[i].item())   # normal EWMA
            else:
                self.tau_hist[i] = self.mu * self.tau_hist[i]  # attestation failure (μ=0.5)

        # Eq (14): composite trust
        tau = (self.a * tau_crypto +
               self.b * tau_stat +
               self.g * self.tau_hist.to(v_flags.device))
        return tau

    def aggregate(self,
                  gradients:   List[torch.Tensor],   # per-client gradient list
                  trust:       torch.Tensor,          # (N,) trust scores
                  v_flags:     torch.Tensor,          # (N,) CGAP pass/fail
                  data_props:  torch.Tensor,          # (N,) p_i data proportions
    ) -> torch.Tensor:
        """Trust-weighted aggregation -- Equations (18)-(19)."""
        verified_ids = (v_flags == 1).nonzero(as_tuple=True)[0].tolist()
        if not verified_ids:
            # Fallback: mean aggregation
            return torch.stack(gradients).mean(dim=0)

        # Normalise trust over verified clients -- Eq (19)
        tau_v = trust[verified_ids] * data_props[verified_ids]
        tau_v = tau_v / (tau_v.sum() + 1e-10)

        # Weighted sum with gradient clipping -- Eq (18)
        agg = torch.zeros_like(gradients[0])
        for rank, cid in enumerate(verified_ids):
            clipped = self._clip(gradients[cid])
            agg += tau_v[rank] * clipped

        return agg

    def effective_clients(self, trust: torch.Tensor, v_flags: torch.Tensor) -> float:
        """Inverse participation ratio -- Equation (20)."""
        tau_v = trust[v_flags == 1]
        if tau_v.numel() == 0:
            return 0.0
        tau_v = tau_v / (tau_v.sum() + 1e-10)
        return 1.0 / (tau_v ** 2).sum().item()


# =============================================================================
# FedSentinel Server  (Algorithm 1)
# =============================================================================

class FedSentinelServer:
    """Central server implementing Algorithm 1 (FedSentinel Core Training).
    Orchestrates CGAP, CADE, DT-RoA across communication rounds.
    """

    def __init__(self,
                 global_model: nn.Module,
                 n_clients:    int = N_CLIENTS,
                 global_lr:    float = 0.01):
        self.model     = global_model
        self.N         = n_clients
        self.eta       = global_lr

        self.cgap   = CGAPModule()
        self.cade   = CADEModule()
        self.dtroa  = DTRoAModule(n_clients)

        self.ref_grad: Optional[torch.Tensor] = None   # g^{t-1}
        self.round    = 0

    def aggregate(self,
                  client_grads_q: List[torch.Tensor],   # quantised gradients from clients
                  data_props:     torch.Tensor,         # (N,) p_i proportions
                  ref_grad:       Optional[torch.Tensor],
                  root_loader,                          # DataLoader for D_root
    ) -> Tuple[torch.Tensor, Dict]:
        """Execute one round of Algorithm 1 (Steps 10-24).

        Returns:
            aggregated_grad: gradient to update global model
            info: diagnostic dict (trust scores, flagged clients, etc.)
        """
        t = self.round
        N = len(client_grads_q)

        # ── Phase 1: CGAP Batch Verification (Steps 11-13) ──────────────────
        v_flags = torch.zeros(N, dtype=torch.long)
        if CGAP_ENABLED:
            verifications = self.cgap.batch_verify(client_grads_q, ref_grad)
            for i, ok in enumerate(verifications):
                v_flags[i] = int(ok)
        else:
            v_flags[:] = 1   # skip CGAP for ablation

        verified_ids = v_flags.nonzero(as_tuple=True)[0].tolist()

        # ── Dequantise for aggregation ────────────────────────────────────────
        grads_float = [self.cgap.dequantize(q) for q in client_grads_q]

        # ── Phase 2: CADE Spectral Detection (Steps 14-20) ───────────────────
        cade_flagged = []
        if CADE_ENABLED and verified_ids:
            G_verified = torch.stack([grads_float[i] for i in verified_ids])
            cade_flagged = self.cade.detect(G_verified, verified_ids, t)

        # ── Statistical trust (cosine similarity with server reference) ────────
        ref_g = self._compute_reference_gradient(root_loader)
        stat_cosines = torch.zeros(N)
        for i, g in enumerate(grads_float):
            cos = F.cosine_similarity(g.unsqueeze(0), ref_g.unsqueeze(0)).item()
            stat_cosines[i] = max(0.0, cos)    # Eq (16): max(0, cosine)

        # ── Phase 3: DT-RoA Trust Update & Aggregation (Steps 21-23) ─────────
        trust = self.dtroa.compute_trust(v_flags, stat_cosines, cade_flagged)
        agg_grad = self.dtroa.aggregate(grads_float, trust, v_flags, data_props)

        # Store for next round reference
        self.ref_grad = agg_grad.detach().clone()

        info = {
            "round":           t,
            "n_verified":      int(v_flags.sum().item()),
            "n_flagged":       len(cade_flagged),
            "trust_mean":      float(trust.mean().item()),
            "trust_min":       float(trust.min().item()),
            "effective_N":     self.dtroa.effective_clients(trust, v_flags),
        }
        self.round += 1
        return agg_grad, info

    @torch.no_grad()
    def _compute_reference_gradient(self, root_loader) -> torch.Tensor:
        """Compute server reference gradient g_ref by running K SGD steps on D_root.
        Section 4.2: g_ref^t = w^t - w_root^{t,K}
        """
        model_copy = copy.deepcopy(self.model)
        opt = torch.optim.SGD(model_copy.parameters(), lr=LOCAL_LR)
        model_copy.train()
        for _ in range(K_LOCAL_EPOCHS):
            for x, y in root_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(model_copy(x), y)
                loss.backward()
                opt.step()
        w_global = get_flat_params(self.model)
        w_root   = get_flat_params(model_copy)
        return (w_global - w_root).to(DEVICE)

    def update_global_model(self, agg_grad: torch.Tensor) -> None:
        """Apply aggregated gradient to global model -- Algorithm 1 Step 24."""
        w = get_flat_params(self.model)
        set_flat_params(self.model, w - self.eta * agg_grad)


# =============================================================================
# FedSentinel Client  (Algorithm 1 Steps 4-9)
# =============================================================================

class FedSentinelClient:
    """FL client implementing local training + CGAP attestation."""

    def __init__(self, client_id: int, local_loader, model: nn.Module):
        self.cid     = client_id
        self.loader  = local_loader
        self.model   = copy.deepcopy(model)
        self.cgap    = CGAPModule()

    def local_train(self,
                    global_params: torch.Tensor,
                    ref_grad: Optional[torch.Tensor] = None,
    ) -> Tuple[bool, torch.Tensor, bytes]:
        """Local training + CGAP attestation.
        Algorithm 1 Steps 4-9.
        Returns (v_i, grad_q, commitment).
        """
        # Load global model
        set_flat_params(self.model, global_params.clone())
        self.model.to(DEVICE).train()
        opt = torch.optim.SGD(self.model.parameters(), lr=LOCAL_LR, momentum=0.9)

        w_before = global_params.clone()

        # K steps of local SGD -- Equation (2)
        for _ in range(K_LOCAL_EPOCHS):
            for x, y in self.loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                loss = F.cross_entropy(self.model(x), y)
                loss.backward()
                opt.step()

        # Gradient update g_i^t = w^t - w_i^{t,K} -- Equation (3)
        w_after = get_flat_params(self.model).cpu()
        grad    = (w_before.cpu() - w_after)

        # CGAP Attestation -- Algorithm 3
        v_i, grad_q, commitment = self.cgap.attest(grad.to(DEVICE), ref_grad)
        return v_i, grad_q, commitment
