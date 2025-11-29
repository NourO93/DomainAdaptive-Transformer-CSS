# train_ft_dann.py
import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC

from data import prepare_cass_data
from models import FTDANN


@dataclass
class TrainConfig:
    data_csv: str = "CASS_Spectrum_Dataset.csv"
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 20
    lambda_grl_init: float = 0.3
    labeled_fraction: float = 0.05
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "./best_dann.pt"


# -----------------------------
# Dataset utilities
# -----------------------------

def to_tensor_dataset(X, y=None):
    X = np.array(X, dtype=np.float32)
    if y is not None:
        y = np.array(y, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return TensorDataset(X_tensor, y_tensor)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        return TensorDataset(X_tensor)


def build_dataloaders(arrays: dict, batch_size: int):
    source_train_ds = to_tensor_dataset(arrays["X_src_train"], arrays["y_src_train"])
    source_val_ds = to_tensor_dataset(arrays["X_src_val"], arrays["y_src_val"])
    target_train_ds = to_tensor_dataset(arrays["X_tgt_train"])  # unlabeled
    target_test_ds = to_tensor_dataset(arrays["X_tgt_test"], arrays["y_tgt_test"])

    src_train_loader = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True)
    src_val_loader = DataLoader(source_val_ds, batch_size=batch_size, shuffle=False)
    tgt_train_loader = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True)
    tgt_test_loader = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False)

    return src_train_loader, src_val_loader, tgt_train_loader, tgt_test_loader


# -----------------------------
# Metrics helper
# -----------------------------

def confusion_metrics(y_true, probs, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    preds_bin = (probs > threshold).astype(int)

    TP = ((preds_bin == 1) & (y_true == 1)).sum()
    TN = ((preds_bin == 0) & (y_true == 0)).sum()
    FP = ((preds_bin == 1) & (y_true == 0)).sum()
    FN = ((preds_bin == 0) & (y_true == 1)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    Pd = TP / (TP + FN + 1e-8)
    Pfa = FP / (FP + TN + 1e-8)
    Pmd = 1.0 - Pd

    return dict(
        accuracy=acc,
        Pd=Pd,
        Pfa=Pfa,
        Pmd=Pmd,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN,
    )


# -----------------------------
# DANN training
# -----------------------------

def train_dann(
    model: FTDANN,
    src_train_loader: DataLoader,
    src_val_loader: DataLoader,
    tgt_train_loader: DataLoader,
    cfg: TrainConfig,
):
    device = cfg.device
    model.to(device)

    criterion_cls = nn.BCELoss()
    criterion_dom = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    best_val_loss = float("inf")
    patience_counter = 0

    num_total_steps = cfg.epochs * min(len(src_train_loader), len(tgt_train_loader))

    step_idx = 0
    for epoch in range(cfg.epochs):
        model.train()
        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)
        num_batches = min(len(src_iter), len(tgt_iter))

        total_cls, total_dom = 0.0, 0.0

        for _ in range(num_batches):
            src_x, src_y = next(src_iter)
            tgt_x = next(tgt_iter)[0]

            src_x, src_y, tgt_x = (
                src_x.to(device),
                src_y.to(device),
                tgt_x.to(device),
            )

            # Dynamic lambda schedule for GRL
            p = step_idx / max(1, num_total_steps - 1)
            lambda_dom = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            model.grl.lambda_ = lambda_dom

            optimizer.zero_grad()

            y_pred, d_src = model(src_x)
            d_tgt = model(tgt_x)[1]

            loss_cls = criterion_cls(y_pred, src_y)

            d_labels_src = torch.zeros_like(d_src)
            d_labels_tgt = torch.ones_like(d_tgt)

            loss_dom = criterion_dom(d_src, d_labels_src) + criterion_dom(
                d_tgt, d_labels_tgt
            )

            loss_total = loss_cls + lambda_dom * loss_dom
            loss_total.backward()
            optimizer.step()

            total_cls += loss_cls.item()
            total_dom += loss_dom.item()
            step_idx += 1

        avg_cls = total_cls / num_batches
        avg_dom = total_dom / num_batches

        # Validation on source
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in src_val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds, _ = model(x_val)
                val_loss += criterion_cls(preds, y_val).item()
        val_loss /= len(src_val_loader)

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] | λ={lambda_dom:.3f} | "
            f"L_cls: {avg_cls:.4f} | L_dom: {avg_dom:.4f} | Val_Loss: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_path)
            print(f"✅ Best model saved at epoch {epoch+1} (Val_Loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"⏹️ Early stopping triggered at epoch {epoch+1}.")
                break

        model.train()

    print(f"\n🎯 Training complete. Best model saved to: {cfg.save_path}")


# -----------------------------
# Evaluation on target
# -----------------------------

def evaluate_on_target(model: FTDANN, tgt_test_loader: DataLoader, device: str):
    model.to(device)
    model.eval()

    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    auc_metric = BinaryAUROC().to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tgt_test_loader:
            x, y = x.to(device), y.to(device)
            preds, _ = model(x)
            acc_metric.update(preds, y)
            f1_metric.update(preds, y)
            auc_metric.update(preds, y)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    auc = auc_metric.compute().item()

    all_preds = torch.cat(all_preds).numpy().ravel()
    all_labels = torch.cat(all_labels).numpy().ravel()

    cm = confusion_metrics(all_labels, all_preds, threshold=0.5)

    print("\n===== Target Domain Evaluation (DANN) =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Pd (Det.): {cm['Pd']:.4f}")
    print(f"Pfa:       {cm['Pfa']:.4f}")
    print(f"Pmd:       {cm['Pmd']:.4f}")

    return acc, f1, auc, cm


# -----------------------------
# Semi-supervised fine-tuning + threshold calibration
# -----------------------------

def fine_tune_and_calibrate(
    model: FTDANN,
    arrays: dict,
    device: str,
    epochs_ft: int = 100,
):
    X_tgt_labeled = np.array(arrays["X_tgt_labeled"], dtype=np.float32)
    y_tgt_labeled = np.array(arrays["y_tgt_labeled"], dtype=np.float32)
    X_tgt_test = np.array(arrays["X_tgt_test"], dtype=np.float32)
    y_tgt_test = np.array(arrays["y_tgt_test"], dtype=np.float32)

    # Optional balancing of positives in tiny labeled subset
    pos_idx = y_tgt_labeled == 1
    neg_idx = y_tgt_labeled == 0
    if pos_idx.sum() > 0 and neg_idx.sum() > 0:
        scale = int(len(neg_idx) / (pos_idx.sum() + 1))
        X_pos = np.repeat(X_tgt_labeled[pos_idx], scale, axis=0)
        y_pos = np.repeat(y_tgt_labeled[pos_idx], scale, axis=0)
        X_tgt_labeled = np.concatenate([X_tgt_labeled, X_pos], axis=0)
        y_tgt_labeled = np.concatenate([y_tgt_labeled, y_pos], axis=0)

    print(
        f"Labeled target set after balancing: {len(y_tgt_labeled)} samples"
    )

    tgt_labeled_ds = TensorDataset(
        torch.tensor(X_tgt_labeled, dtype=torch.float32),
        torch.tensor(y_tgt_labeled, dtype=torch.float32).view(-1, 1),
    )
    tgt_labeled_loader = DataLoader(
        tgt_labeled_ds, batch_size=16, shuffle=True
    )

    # Copy DANN weights
    ft_model = deepcopy(model).to(device)
    for param in ft_model.parameters():
        param.requires_grad = True

    ft_optimizer = optim.Adam(
        ft_model.parameters(), lr=1e-4, weight_decay=1e-5
    )
    ft_criterion = nn.BCELoss()

    # Fine-tune
    ft_model.train()
    for epoch in range(epochs_ft):
        total_loss = 0.0
        for x, y in tgt_labeled_loader:
            x, y = x.to(device), y.to(device)
            preds, _ = ft_model(x)
            loss = ft_criterion(preds, y)
            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(tgt_labeled_loader)
        print(f"[FT Epoch {epoch+1}/{epochs_ft}] Loss {avg_loss:.4f}")

    # Predict on target test set
    ft_model.eval()
    with torch.no_grad():
        preds, _ = ft_model(
            torch.tensor(X_tgt_test, dtype=torch.float32).to(device)
        )
    probs = preds.cpu().numpy().ravel()
    y_true = y_tgt_test

    # Threshold calibration by F1
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    f1s = [f1_score(y_true, probs > t) for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    print(f"\nOptimal threshold for F1: {best_t:.3f}")

    cm = confusion_metrics(y_true, probs, threshold=best_t)
    f1_best = f1_score(y_true, probs > best_t)

    print(
        "\n===== Fine-Tuned Target Evaluation (DANN + FT + Calibration) ====="
    )
    print(f"Accuracy:  {cm['accuracy']:.4f}")
    print(f"F1-Score:  {f1_best:.4f}")
    print(f"Pd (Det.): {cm['Pd']:.4f}")
    print(f"Pfa:       {cm['Pfa']:.4f}")
    print(f"Pmd:       {cm['Pmd']:.4f}")

    return ft_model, best_t, cm, f1_best


# -----------------------------
# Main script
# -----------------------------

def main():
    cfg = TrainConfig()
    print(f"Using device: {cfg.device}")
    print(f"Loading data from: {cfg.data_csv}")

    arrays = prepare_cass_data(
        cfg.data_csv,
        labeled_fraction=cfg.labeled_fraction,
        seed=cfg.seed,
    )

    src_train_loader, src_val_loader, tgt_train_loader, tgt_test_loader = (
        build_dataloaders(arrays, cfg.batch_size)
    )

    model = FTDANN(
        input_dim=arrays["input_dim"], lambda_grl=cfg.lambda_grl_init
    )

    # Train DANN
    train_dann(
        model,
        src_train_loader,
        src_val_loader,
        tgt_train_loader,
        cfg,
    )

    # Load best checkpoint
    model.load_state_dict(torch.load(cfg.save_path, map_location=cfg.device))

    # Evaluate base DANN
    evaluate_on_target(model, tgt_test_loader, cfg.device)

    # Fine-tune + calibration
    fine_tune_and_calibrate(model, arrays, cfg.device, epochs_ft=100)


if __name__ == "__main__":
    main()
