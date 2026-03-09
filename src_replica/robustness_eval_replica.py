"""Adversarial robustness evaluation -- P3.

Evaluates the light model (TinyHybridStudent) against:
  1. Gaussian noise (baseline)
  2. FGSM (Fast Gradient Sign Method) -- white-box, single-step
  3. PGD  (Projected Gradient Descent) -- white-box, multi-step

Realistic perturbation budgets are used:
  * CAN features are normalised ~[0,1], so epsilon in {0.01, 0.05, 0.1, 0.2}
  * ETH images are normalised ~[0,1.5], same epsilon range applies

Both targeted (flip label) and untargeted attacks are evaluated.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src_replica.dataloader_correlated_replica import CorrelatedHybridVehicleDataset
from src_replica.runtime.standards import CAN_WINDOW_SIZE_STANDARD, ETH_WINDOW_SIZE_STANDARD

sys.path.insert(0, BASE_DIR)
from architecture_improved import TinyHybridStudent

CAN_FEATURES_16 = [
    "CAN_ID", "DLC", "D0", "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "can_id_freq_global", "can_id_freq_win", "payload_entropy",
    "inter_arrival", "inter_arrival_roll_mean", "id_switch_rate_win",
]


# ---------------------------------------------------------------------------
# Attack implementations
# ---------------------------------------------------------------------------

def fgsm_attack(
    model: nn.Module,
    can_input: torch.Tensor,
    eth_input: torch.Tensor,
    label: int,
    epsilon: float,
    targeted: bool = False,
) -> tuple:
    """Single-step FGSM; returns perturbed (can, eth)."""
    can_adv = can_input.clone().detach().requires_grad_(True)
    eth_adv = eth_input.clone().detach().requires_grad_(True)

    logits = model(can_adv, eth_adv)
    if targeted:
        # Targeted: minimize loss w.r.t. the *wrong* class (flip label)
        target_label = 1 - label
        target_tensor = torch.tensor([target_label], dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, target_tensor)
        loss.backward()
        # Step to DECREASE loss w.r.t wrong target
        can_adv = (can_input - epsilon * can_adv.grad.sign()).detach()
        eth_adv = (eth_input - epsilon * eth_adv.grad.sign()).detach()
    else:
        # Untargeted: maximize loss w.r.t. true label
        target_tensor = torch.tensor([label], dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, target_tensor)
        loss.backward()
        can_adv = (can_input + epsilon * can_adv.grad.sign()).detach()
        eth_adv = (eth_input + epsilon * eth_adv.grad.sign()).detach()
    return can_adv, eth_adv


def pgd_attack(
    model: nn.Module,
    can_input: torch.Tensor,
    eth_input: torch.Tensor,
    label: int,
    epsilon: float,
    alpha: float = 0.01,
    num_steps: int = 10,
    targeted: bool = False,
) -> tuple:
    """Multi-step PGD; returns perturbed (can, eth)."""
    can_orig = can_input.clone().detach()
    eth_orig = eth_input.clone().detach()

    can_adv = can_orig.clone()
    eth_adv = eth_orig.clone()
    target_label = (1 - label) if targeted else label

    for _ in range(num_steps):
        can_adv = can_adv.detach().requires_grad_(True)
        eth_adv = eth_adv.detach().requires_grad_(True)

        logits = model(can_adv, eth_adv)
        target_tensor = torch.tensor([target_label], dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, target_tensor)
        loss.backward()

        # Targeted: minimize loss w.r.t wrong class; Untargeted: maximize loss w.r.t true class
        sign = -1.0 if targeted else 1.0
        can_adv = (can_adv + sign * alpha * can_adv.grad.sign()).detach()
        eth_adv = (eth_adv + sign * alpha * eth_adv.grad.sign()).detach()

        # Project back into epsilon-ball around original
        can_adv = torch.clamp(can_adv, can_orig - epsilon, can_orig + epsilon)
        eth_adv = torch.clamp(eth_adv, eth_orig - epsilon, eth_orig + epsilon)

    return can_adv, eth_adv


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    ds: CorrelatedHybridVehicleDataset,
    attack_fn,
    epsilon: float,
    max_samples: int,
    **attack_kwargs,
) -> Dict:
    """Run *attack_fn* on each sample and return accuracy metrics."""
    model.eval()
    y_true: List[int] = []
    y_pred_clean: List[int] = []
    y_pred_adv: List[int] = []

    for i in range(min(max_samples, len(ds))):
        sample = ds[i]
        can_t = sample["can"].unsqueeze(0)
        eth_t = sample["eth"].unsqueeze(0)
        label = int(sample["label"].item())
        y_true.append(label)

        # Clean prediction
        with torch.no_grad():
            logits_clean = model(can_t, eth_t)
            pred_clean = int(logits_clean.argmax(dim=1).item())
        y_pred_clean.append(pred_clean)

        # Adversarial prediction
        if attack_fn is None:
            # No attack -- just Gaussian noise
            sigma = epsilon
            can_adv = can_t + torch.randn_like(can_t) * sigma
            eth_adv = eth_t + torch.randn_like(eth_t) * sigma
        else:
            can_adv, eth_adv = attack_fn(
                model, can_t, eth_t, label, epsilon=epsilon, **attack_kwargs
            )

        with torch.no_grad():
            logits_adv = model(can_adv, eth_adv)
            pred_adv = int(logits_adv.argmax(dim=1).item())
        y_pred_adv.append(pred_adv)

    y_true = np.asarray(y_true, dtype=int)
    y_pred_clean = np.asarray(y_pred_clean, dtype=int)
    y_pred_adv = np.asarray(y_pred_adv, dtype=int)

    clean_acc = float(np.mean(y_true == y_pred_clean))
    adv_acc = float(np.mean(y_true == y_pred_adv))
    flip_rate = float(np.mean(y_pred_clean != y_pred_adv))  # % predictions changed by attack
    n_attack = int(np.sum(y_true == 1))
    n_normal = int(np.sum(y_true == 0))

    # Per-class: attack evasion rate and false trigger rate
    evasion_rate = None
    false_trigger_rate = None
    if n_attack > 0:
        evasion_rate = float(np.mean(y_pred_adv[y_true == 1] == 0))  # attacks misclassified as normal
    if n_normal > 0:
        false_trigger_rate = float(np.mean(y_pred_adv[y_true == 0] == 1))  # normal misclassified as attack

    return {
        "epsilon": epsilon,
        "samples": int(len(y_true)),
        "clean_accuracy": clean_acc,
        "adversarial_accuracy": adv_acc,
        "accuracy_drop": clean_acc - adv_acc,
        "flip_rate": flip_rate,
        "evasion_rate": evasion_rate,
        "false_trigger_rate": false_trigger_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial robustness evaluation (P3).")
    parser.add_argument("--light_model", default="models/student_tiny_improved.pth")
    parser.add_argument("--can_csv", default="datasets/replica_can_b1_engineered/can_dos_train.csv")
    parser.add_argument("--eth_csv", default="datasets/replica_eth_smoke/eth_driving_01_injected_replica_packets.csv")
    parser.add_argument("--eth_npy", default="datasets/eth_driving_01_injected_images-003.npy")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", default="reports/robustness_report.json")
    args = parser.parse_args()

    # Load model
    model = TinyHybridStudent(input_dim=16, hidden_dim=64, num_classes=2)
    ckpt = torch.load(args.light_model, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Load dataset
    ds = CorrelatedHybridVehicleDataset(
        can_csv_path=args.can_csv,
        eth_packet_csv_path=args.eth_csv,
        eth_npy_path=args.eth_npy,
        can_features=CAN_FEATURES_16,
        can_window_size=CAN_WINDOW_SIZE_STANDARD,
        eth_window_size=ETH_WINDOW_SIZE_STANDARD,
        eth_overlap=0,
    )
    print(f"Dataset: {len(ds)} samples (max_samples={args.max_samples})")

    # Perturbation budgets (realistic for normalised features)
    epsilons = [0.0, 0.01, 0.05, 0.1, 0.2]

    # 1. Gaussian noise baseline
    print("\n=== Gaussian Noise ===")
    gauss_rows = []
    for eps in epsilons:
        r = evaluate(model, ds, attack_fn=None, epsilon=eps, max_samples=args.max_samples)
        r["method"] = "gaussian"
        gauss_rows.append(r)
        print(f"  sigma={eps:.3f}  clean={r['clean_accuracy']:.2%}  adv={r['adversarial_accuracy']:.2%}  drop={r['accuracy_drop']:.2%}")

    # 2. FGSM (untargeted)
    print("\n=== FGSM (untargeted) ===")
    fgsm_rows = []
    for eps in epsilons[1:]:  # skip 0
        r = evaluate(model, ds, attack_fn=fgsm_attack, epsilon=eps, max_samples=args.max_samples, targeted=False)
        r["method"] = "fgsm_untargeted"
        fgsm_rows.append(r)
        print(f"  eps={eps:.3f}  clean={r['clean_accuracy']:.2%}  adv={r['adversarial_accuracy']:.2%}  drop={r['accuracy_drop']:.2%}  flip={r['flip_rate']:.2%}")

    # 3. FGSM (targeted – try to flip label)
    print("\n=== FGSM (targeted) ===")
    fgsm_tgt_rows = []
    for eps in epsilons[1:]:
        r = evaluate(model, ds, attack_fn=fgsm_attack, epsilon=eps, max_samples=args.max_samples, targeted=True)
        r["method"] = "fgsm_targeted"
        fgsm_tgt_rows.append(r)
        print(f"  eps={eps:.3f}  clean={r['clean_accuracy']:.2%}  adv={r['adversarial_accuracy']:.2%}  drop={r['accuracy_drop']:.2%}  flip={r['flip_rate']:.2%}")

    # 4. PGD (untargeted, 10 steps)
    print("\n=== PGD-10 (untargeted) ===")
    pgd_rows = []
    for eps in epsilons[1:]:
        alpha = eps / 4.0  # Step size = eps/4
        r = evaluate(model, ds, attack_fn=pgd_attack, epsilon=eps, max_samples=args.max_samples,
                     targeted=False, alpha=alpha, num_steps=10)
        r["method"] = "pgd10_untargeted"
        pgd_rows.append(r)
        print(f"  eps={eps:.3f}  clean={r['clean_accuracy']:.2%}  adv={r['adversarial_accuracy']:.2%}  drop={r['accuracy_drop']:.2%}  flip={r['flip_rate']:.2%}")

    # 5. PGD (targeted, 20 steps – strongest attack)
    print("\n=== PGD-20 (targeted) ===")
    pgd_tgt_rows = []
    for eps in epsilons[1:]:
        alpha = eps / 5.0
        r = evaluate(model, ds, attack_fn=pgd_attack, epsilon=eps, max_samples=args.max_samples,
                     targeted=True, alpha=alpha, num_steps=20)
        r["method"] = "pgd20_targeted"
        pgd_tgt_rows.append(r)
        print(f"  eps={eps:.3f}  clean={r['clean_accuracy']:.2%}  adv={r['adversarial_accuracy']:.2%}  drop={r['accuracy_drop']:.2%}  flip={r['flip_rate']:.2%}")

    # Assemble report
    all_rows = gauss_rows + fgsm_rows + fgsm_tgt_rows + pgd_rows + pgd_tgt_rows
    clean_acc = gauss_rows[0]["clean_accuracy"]  # eps=0

    # Find worst-case accuracy across all attacks
    worst_adv_acc = min(r["adversarial_accuracy"] for r in all_rows if r["epsilon"] > 0)
    max_drop = max(r["accuracy_drop"] for r in all_rows if r["epsilon"] > 0)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "type": "adversarial_robustness_eval",
        "model": args.light_model,
        "dataset": args.can_csv,
        "max_samples": args.max_samples,
        "clean_accuracy": clean_acc,
        "worst_case_adversarial_accuracy": worst_adv_acc,
        "max_accuracy_drop": max_drop,
        "perturbation_budgets": epsilons[1:],
        "methods_tested": ["gaussian", "fgsm_untargeted", "fgsm_targeted", "pgd10_untargeted", "pgd20_targeted"],
        "results": all_rows,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Clean accuracy:     {clean_acc:.2%}")
    print(f"Worst-case adv acc: {worst_adv_acc:.2%}")
    print(f"Max accuracy drop:  {max_drop:.2%}")
    print(f"Report saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
