# =============================================================================
# train.py -- Full FedSentinel training loop
# Paper Section 6: 500 communication rounds, 100 clients, SGD momentum 0.9
# Algorithm 1 orchestration with CGAP + CADE + DT-RoA
# =============================================================================

import os, copy, time, random, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from config import (
    DATASET, N_CLIENTS, N_ROUNDS, K_LOCAL_EPOCHS, BATCH_SIZE,
    GLOBAL_LR, LOCAL_LR, MOMENTUM, LR_SCHEDULER, SEED,
    BYZANTINE_FRACTION, ATTACK_TYPE,
    CHECKPOINT_BEST, CHECKPOINT_LAST, RESULTS_DIR, TENSORBOARD_DIR,
    EVAL_EVERY, LOG_EVERY, SAVE_EVERY, DEVICE,
    RANDOM_SEEDS, MODEL_ARCH, NUM_CLASSES,
)
from dataset import get_dataloaders
from model import build_model, get_flat_params, set_flat_params, apply_attack
from fedsentinel import FedSentinelServer, FedSentinelClient
from utils import (set_seed, compute_accuracy, AverageMeter,
                   save_checkpoint, load_checkpoint)


def build_lr_scheduler(optimizer, n_rounds: int, scheduler_type: str = LR_SCHEDULER):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_rounds)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    return None


def run_fedsentinel(seed: int = SEED):
    """Single FL run for one seed."""
    set_seed(seed)
    print(f"\n{'='*60}")
    print(f" FedSentinel Training  |  Dataset: {DATASET}  |  Seed: {seed}")
    print(f" Attack: {ATTACK_TYPE}  |  Byzantine: {BYZANTINE_FRACTION*100:.0f}%")
    print(f" Rounds: {N_ROUNDS}  |  Clients: {N_CLIENTS}  |  K: {K_LOCAL_EPOCHS}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    client_loaders, val_loader, test_loader, root_ds = get_dataloaders()
    root_loader = torch.utils.data.DataLoader(
        root_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # ── Model & Server ────────────────────────────────────────────────────────
    global_model = build_model(MODEL_ARCH, NUM_CLASSES)
    server = FedSentinelServer(global_model, n_clients=N_CLIENTS,
                               global_lr=GLOBAL_LR)

    # Dummy optimiser for LR scheduling only (actual optimisation is client-side)
    dummy_opt = torch.optim.SGD(global_model.parameters(), lr=GLOBAL_LR,
                                momentum=MOMENTUM)
    scheduler = build_lr_scheduler(dummy_opt, N_ROUNDS)

    # ── Byzantine setup ───────────────────────────────────────────────────────
    n_byzantine  = int(BYZANTINE_FRACTION * N_CLIENTS)
    byzantine_ids = set(random.sample(range(N_CLIENTS), n_byzantine))
    honest_ids    = [i for i in range(N_CLIENTS) if i not in byzantine_ids]
    print(f"Byzantine clients ({n_byzantine}): {sorted(byzantine_ids)}")

    # Data proportions p_i (uniform for simplicity; weighted by shard size in paper)
    data_props = torch.ones(N_CLIENTS) / N_CLIENTS

    # ── TensorBoard ───────────────────────────────────────────────────────────
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(TENSORBOARD_DIR, f"seed_{seed}"))

    best_val_acc = 0.0
    history = []

    # ── Communication Rounds (Algorithm 1) ────────────────────────────────────
    for t in range(N_ROUNDS):
        round_start = time.time()

        # Server broadcasts w^t
        global_params = get_flat_params(global_model).detach().clone()

        # ── Client local training (Algorithm 1 Steps 4-9) ─────────────────
        client_grads_q = []
        honest_grads   = []    # for coordinated attacks: need honest gradients

        for cid in range(N_CLIENTS):
            client = FedSentinelClient(cid, client_loaders[cid], global_model)
            v_i, grad_q, commitment = client.local_train(
                global_params, server.ref_grad)
            client_grads_q.append(grad_q)
            if cid not in byzantine_ids:
                honest_grads.append(server.cgap.dequantize(grad_q))

        # ── Inject Byzantine attacks ───────────────────────────────────────
        if honest_grads:
            honest_tensor = torch.stack(honest_grads)   # (N_honest, d)
        else:
            honest_tensor = None

        for byz_id in byzantine_ids:
            byz_grad = server.cgap.dequantize(client_grads_q[byz_id])
            if honest_tensor is not None:
                poisoned_tensor = apply_attack(
                    gradients=byz_grad.unsqueeze(0),
                    honest_gradients=honest_tensor,
                    attack_type=ATTACK_TYPE,
                    round_idx=t,
                    total_rounds=N_ROUNDS,
                )
                poisoned = poisoned_tensor[0]
            else:
                poisoned = byz_grad

            # Re-quantise poisoned gradient
            client_grads_q[byz_id] = server.cgap.quantize(poisoned)

        # ── Server aggregation (Algorithm 1 Steps 10-24) ──────────────────
        agg_grad, info = server.aggregate(
            client_grads_q, data_props, server.ref_grad, root_loader)

        server.update_global_model(agg_grad)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = GLOBAL_LR

        round_time = time.time() - round_start

        # ── Evaluation ─────────────────────────────────────────────────────
        if (t + 1) % EVAL_EVERY == 0 or t == 0:
            val_acc  = compute_accuracy(global_model, val_loader)
            test_acc = compute_accuracy(global_model, test_loader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint({"round": t, "model_state": global_model.state_dict(),
                                 "val_acc": val_acc, "test_acc": test_acc},
                                CHECKPOINT_BEST)

            writer.add_scalar("Accuracy/val",     val_acc,  t)
            writer.add_scalar("Accuracy/test",    test_acc, t)
            writer.add_scalar("Trust/mean",       info["trust_mean"], t)
            writer.add_scalar("Trust/n_verified", info["n_verified"],  t)
            writer.add_scalar("CADE/n_flagged",   info["n_flagged"],  t)
            writer.add_scalar("LR",               current_lr,         t)

            record = {"round": t + 1, "val_acc": val_acc, "test_acc": test_acc,
                      "lr": current_lr, **info, "round_time": round_time}
            history.append(record)

            print(f"Round {t+1:4d}/{N_ROUNDS} | "
                  f"Val: {val_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Verified: {info['n_verified']}/{N_CLIENTS} | "
                  f"Flagged: {info['n_flagged']} | "
                  f"Trust: {info['trust_mean']:.3f} | "
                  f"Time: {round_time:.1f}s")

        elif (t + 1) % LOG_EVERY == 0:
            print(f"Round {t+1:4d}/{N_ROUNDS} | "
                  f"Verified: {info['n_verified']}/{N_CLIENTS} | "
                  f"Time: {round_time:.2f}s")

        # Save periodic checkpoint
        if (t + 1) % SAVE_EVERY == 0:
            save_checkpoint({"round": t, "model_state": global_model.state_dict(),
                             "val_acc": best_val_acc},
                            CHECKPOINT_LAST)

    # ── Final evaluation ──────────────────────────────────────────────────────
    test_acc = compute_accuracy(global_model, test_loader)
    print(f"\n[FINAL] Test Accuracy: {test_acc:.2f}%  (Best Val: {best_val_acc:.2f}%)")

    # Save history
    save_checkpoint({"round": N_ROUNDS, "model_state": global_model.state_dict(),
                     "test_acc": test_acc}, CHECKPOINT_LAST)

    hist_path = os.path.join(RESULTS_DIR, f"history_seed{seed}.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[SAVED] {hist_path}")

    writer.close()
    return test_acc, history


def main():
    """Run 3-seed experiment and report mean ± std (Table 4: seeds 42, 123, 456)."""
    all_acc = []
    for seed in RANDOM_SEEDS:
        final_acc, _ = run_fedsentinel(seed=seed)
        all_acc.append(final_acc)

    mean_acc = np.mean(all_acc)
    std_acc  = np.std(all_acc)
    print(f"\n{'='*60}")
    print(f" Final Results (3-seed average)")
    print(f" Dataset: {DATASET}  |  Attack: {ATTACK_TYPE}  |  Byzantine: {BYZANTINE_FRACTION}")
    print(f" Test Accuracy: {mean_acc:.2f} ± {std_acc:.2f}%")
    print(f"{'='*60}")

    # Save summary
    summary = {
        "dataset": DATASET, "attack": ATTACK_TYPE,
        "byzantine_fraction": BYZANTINE_FRACTION,
        "seeds": RANDOM_SEEDS, "accuracies": all_acc,
        "mean": mean_acc, "std": std_acc,
    }
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {summary_path}")


if __name__ == "__main__":
    main()
