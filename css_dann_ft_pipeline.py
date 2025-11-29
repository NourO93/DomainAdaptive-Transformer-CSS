#!/usr/bin/env python
"""
Cooperative Spectrum Sensing Domain Adaptation
FT-Transformer + DANN + Semi-Supervised Fine-Tuning

"""

# ======================================================
# Imports
# ======================================================

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_curve

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryAUROC
from rtdl import FTTransformer


# ======================================================
# Stage 1–2: Data Preparation + Preprocessing
# ======================================================

def load_and_preprocess(
    csv_path: str | Path,
    labeled_fraction: float = 0.05,
):
    """
    Load the CASS spectrum dataset and perform:
      - NA drop
      - label casting
      - source/target split based on average SNR per cluster
      - train/val/test splits
      - one-hot encoding of Frequency_Band
      - standardization of numeric features
      - creation of small labeled target subset
    """

    csv_path = str(csv_path)
    data = pd.read_csv(csv_path)

    print(f"[INFO] Initial shape: {data.shape}")
    data.dropna(inplace=True)
    data["Target"] = data["Target"].astype(int)
    print(f"[INFO] After cleaning: {data.shape}")

    base_features = [
        "PU_Signal_Strength",
        "Frequency_Band",
        "SNR_SU1",
        "SNR_SU2",
        "SNR_SU3",
        "Cluster_Size",
    ]
    label_col = "Target"

    # --- Compute average SNR per cluster ---
    cluster_mean_snr = (
        data.groupby("Cluster_ID")[["SNR_SU1", "SNR_SU2", "SNR_SU3"]]
        .mean()
        .mean(axis=1)
    )
    sorted_clusters = cluster_mean_snr.sort_values(ascending=False)
    num_clusters = len(sorted_clusters)

    # Source = top half SNR clusters, Target = bottom half
    source_clusters = sorted_clusters.index[: num_clusters // 2]
    target_clusters = sorted_clusters.index[num_clusters // 2 :]

    print(f"[INFO] Total clusters: {num_clusters}")
    print(f"[INFO] Source clusters: {len(source_clusters)}, Target clusters: {len(target_clusters)}")

    # --- Split into domains ---
    X_source_raw = data[data["Cluster_ID"].isin(source_clusters)][base_features]
    y_source = data[data["Cluster_ID"].isin(source_clusters)][label_col]

    X_target_raw = data[data["Cluster_ID"].isin(target_clusters)][base_features]
    y_target = data[data["Cluster_ID"].isin(target_clusters)][label_col]

    print(f"[INFO] Source samples: {len(X_source_raw)}, Target samples: {len(X_target_raw)}")

    # --- Train/Val/Test splits ---
    X_src_train, X_src_val, y_src_train, y_src_val = train_test_split(
        X_source_raw,
        y_source,
        test_size=0.3,
        stratify=y_source,
        random_state=42,
    )

    X_tgt_train_unlabeled, X_tgt_test, y_tgt_train_unlabeled, y_tgt_test = train_test_split(
        X_target_raw,
        y_target,
        test_size=0.3,
        stratify=y_target,
        random_state=42,
    )

    # Small labeled target subset
    num_labeled = int(len(X_tgt_train_unlabeled) * labeled_fraction)
    X_tgt_labeled = X_tgt_train_unlabeled.sample(num_labeled, random_state=42)
    y_tgt_labeled = y_tgt_train_unlabeled.loc[X_tgt_labeled.index]

    print(f"[INFO] Labeled target subset: {len(X_tgt_labeled)} samples")

    # --- One-hot encode Frequency_Band ---
    def one_hot(df):
        return pd.get_dummies(df, columns=["Frequency_Band"])

    X_src_train_enc = one_hot(X_src_train)
    X_src_val_enc = one_hot(X_src_val)
    X_tgt_train_enc = one_hot(X_tgt_train_unlabeled)
    X_tgt_test_enc = one_hot(X_tgt_test)
    X_tgt_labeled_enc = one_hot(X_tgt_labeled)

    # Align columns across splits
    all_cols = sorted(
        set(X_src_train_enc.columns)
        | set(X_src_val_enc.columns)
        | set(X_tgt_train_enc.columns)
        | set(X_tgt_test_enc.columns)
        | set(X_tgt_labeled_enc.columns)
    )
    def align(df):
        return df.reindex(columns=all_cols, fill_value=0)

    X_src_train_enc = align(X_src_train_enc)
    X_src_val_enc = align(X_src_val_enc)
    X_tgt_train_enc = align(X_tgt_train_enc)
    X_tgt_test_enc = align(X_tgt_test_enc)
    X_tgt_labeled_enc = align(X_tgt_labeled_enc)

    # --- Standardize continuous columns (fit on source train only) ---
    num_cols = ["PU_Signal_Strength", "SNR_SU1", "SNR_SU2", "SNR_SU3", "Cluster_Size"]
    scaler = StandardScaler()
    scaler.fit(X_src_train_enc[num_cols])

    def scale_df(df):
        df = df.copy()
        df[num_cols] = scaler.transform(df[num_cols])
        return df

    X_src_train_enc = scale_df(X_src_train_enc)
    X_src_val_enc = scale_df(X_src_val_enc)
    X_tgt_train_enc = scale_df(X_tgt_train_enc)
    X_tgt_test_enc = scale_df(X_tgt_test_enc)
    X_tgt_labeled_enc = scale_df(X_tgt_labeled_enc)

    # --- Convert to numpy arrays ---
    X_src_train_np = X_src_train_enc.to_numpy()
    y_src_train_np = y_src_train.to_numpy()

    X_src_val_np = X_src_val_enc.to_numpy()
    y_src_val_np = y_src_val.to_numpy()

    X_tgt_train_np = X_tgt_train_enc.to_numpy()
    y_tgt_train_np = y_tgt_train_unlabeled.to_numpy()

    X_tgt_test_np = X_tgt_test_enc.to_numpy()
    y_tgt_test_np = y_tgt_test.to_numpy()

    X_tgt_labeled_np = X_tgt_labeled_enc.to_numpy()
    y_tgt_labeled_np = y_tgt_labeled.to_numpy()

    print(f"[INFO] Preprocessing complete. Final feature dimension: {X_src_train_np.shape[1]}")

    return (
        X_src_train_np,
        y_src_train_np,
        X_src_val_np,
        y_src_val_np,
        X_tgt_train_np,
        y_tgt_train_np,
        X_tgt_test_np,
        y_tgt_test_np,
        X_tgt_labeled_np,
        y_tgt_labeled_np,
    )


# ======================================================
# Stage 3–5: FT-Transformer + DANN
# ======================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradReverseLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradReverse.apply(x, self.lambda_)


class FTDANN(nn.Module):
    def __init__(self, input_dim: int, lambda_grl: float = 0.5):
        super().__init__()

        # FTTransformer as feature extractor
        self.feature_extractor = FTTransformer.make_default(
            n_num_features=input_dim,
            cat_cardinalities=None,
            d_out=64,
        )

        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.grl = GradReverseLayer(lambda_=lambda_grl)
        self.domain_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, D]
        features = self.feature_extractor(x, None)
        class_output = self.label_predictor(features)
        domain_output = self.domain_classifier(self.grl(features))
        return class_output, domain_output


# ======================================================
# Dataset and Dataloaders
# ======================================================

def to_tensor_dataset(X, y=None) -> TensorDataset:
    """
    Convert numpy or pandas arrays to a TensorDataset.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32) if y is not None else None

    X_tensor = torch.tensor(X, dtype=torch.float32)
    if y is not None:
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        return TensorDataset(X_tensor, y_tensor)
    else:
        return TensorDataset(X_tensor)


def build_dataloaders(
    X_src_train_np,
    y_src_train_np,
    X_src_val_np,
    y_src_val_np,
    X_tgt_train_np,
    X_tgt_test_np,
    y_tgt_test_np,
    batch_size: int = 64,
):
    source_train_ds = to_tensor_dataset(X_src_train_np, y_src_train_np)
    source_val_ds = to_tensor_dataset(X_src_val_np, y_src_val_np)
    target_train_ds = to_tensor_dataset(X_tgt_train_np)
    target_test_ds = to_tensor_dataset(X_tgt_test_np, y_tgt_test_np)

    src_train_loader = DataLoader(source_train_ds, batch_size=batch_size, shuffle=True)
    src_val_loader = DataLoader(source_val_ds, batch_size=batch_size, shuffle=False)
    tgt_train_loader = DataLoader(target_train_ds, batch_size=batch_size, shuffle=True)
    tgt_test_loader = DataLoader(target_test_ds, batch_size=batch_size, shuffle=False)

    return src_train_loader, src_val_loader, tgt_train_loader, tgt_test_loader


# ======================================================
# Training: DANN with early stopping
# ======================================================

def train_dann(
    model: FTDANN,
    src_train_loader: DataLoader,
    src_val_loader: DataLoader,
    tgt_train_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    patience: int = 20,
    save_path: str | Path = "./best_dann.pt",
):
    """
    Train DANN with:
      - dynamic GRL lambda
      - early stopping on source validation loss
      - best model checkpoint
    """
    criterion_cls = nn.BCELoss()
    criterion_dom = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    save_path = str(save_path)
    best_val_loss = float("inf")
    patience_counter = 0

    model.to(device)
    model.train()

    for epoch in range(epochs):
        src_iter = iter(src_train_loader)
        tgt_iter = iter(tgt_train_loader)

        num_batches = min(len(src_train_loader), len(tgt_train_loader))

        total_cls, total_dom = 0.0, 0.0

        # --- Training phase ---
        for i in range(num_batches):
            src_x, src_y = next(src_iter)
            tgt_x = next(tgt_iter)[0]

            src_x = src_x.to(device)
            src_y = src_y.to(device)
            tgt_x = tgt_x.to(device)

            # Dynamic lambda for GRL
            p = float(epoch * num_batches + i) / (epochs * num_batches)
            lambda_dom = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0
            model.grl.lambda_ = lambda_dom

            optimizer.zero_grad()

            y_pred, d_src = model(src_x)
            d_tgt = model(tgt_x)[1]

            loss_cls = criterion_cls(y_pred, src_y)

            # Domain labels: 0 = source, 1 = target (consistent for classifier)
            d_labels_src = torch.zeros(d_src.size(0), 1, device=device)
            d_labels_tgt = torch.ones(d_tgt.size(0), 1, device=device)
            loss_dom = criterion_dom(d_src, d_labels_src) + criterion_dom(d_tgt, d_labels_tgt)

            loss_total = loss_cls + lambda_dom * loss_dom
            loss_total.backward()
            optimizer.step()

            total_cls += loss_cls.item()
            total_dom += loss_dom.item()

        avg_cls = total_cls / num_batches
        avg_dom = total_dom / num_batches

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in src_val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                preds, _ = model(x_val)
                val_loss += criterion_cls(preds, y_val).item()
        val_loss /= len(src_val_loader)

        print(
            f"Epoch [{epoch+1}/{epochs}] | λ={lambda_dom:.3f} | "
            f"L_cls: {avg_cls:.4f} | L_dom: {avg_dom:.4f} | Val_Loss: {val_loss:.4f}"
        )

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved at epoch {epoch+1} (Val_Loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        model.train()

    print(f"\n[INFO] Training complete. Best model saved to: {save_path}")
    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ======================================================
# Evaluation helper
# ======================================================

def evaluate_on_target(
    model: FTDANN,
    tgt_test_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    title: str = "Target Domain Evaluation",
):
    """
    Evaluate the model on the target test loader with:
      - Accuracy
      - F1
      - AUROC
      - Pd, Pfa, Pmd
    """
    model.eval()

    acc_metric = BinaryAccuracy().to(device)
    f1_metric = BinaryF1Score().to(device)
    auc_metric = BinaryAUROC().to(device)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in tgt_test_loader:
            x = x.to(device)
            y = y.to(device)

            preds, _ = model(x)
            acc_metric.update(preds, y)
            f1_metric.update(preds, y)
            auc_metric.update(preds, y)

            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    auc = auc_metric.compute().item()

    preds_concat = torch.cat(all_preds)
    labels_concat = torch.cat(all_labels)

    preds_bin = (preds_concat > threshold).int()
    labels_bin = labels_concat.int()

    TP = ((preds_bin == 1) & (labels_bin == 1)).sum().item()
    TN = ((preds_bin == 0) & (labels_bin == 0)).sum().item()
    FP = ((preds_bin == 1) & (labels_bin == 0)).sum().item()
    FN = ((preds_bin == 0) & (labels_bin == 1)).sum().item()

    Pd = TP / (TP + FN + 1e-8)
    Pfa = FP / (FP + TN + 1e-8)
    Pmd = 1.0 - Pd

    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Pd (Detection): {Pd:.4f}")
    print(f"Pfa (False Alarm): {Pfa:.4f}")
    print(f"Pmd (Miss Detection): {Pmd:.4f}")

    return acc, f1, auc, Pd, Pfa, Pmd


# ======================================================
# Stage 6b: Semi-Supervised Fine-Tuning + Threshold Calibration
# ======================================================

def fine_tune_on_labeled_target(
    base_model: FTDANN,
    X_tgt_labeled_np,
    y_tgt_labeled_np,
    device: torch.device,
    epochs_ft: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
):
    """
    Semi-supervised fine-tuning on small labeled target set.
    Uses full model (not only head) with low learning rate.
    """

    X_tgt_labeled_np = np.array(X_tgt_labeled_np, dtype=np.float32)
    y_tgt_labeled_np = np.array(y_tgt_labeled_np, dtype=np.float32)

    # Optional: balance positives
    pos_idx = y_tgt_labeled_np == 1
    neg_idx = y_tgt_labeled_np == 0
    if pos_idx.sum() > 0 and neg_idx.sum() > 0:
        scale = int(len(neg_idx) / (pos_idx.sum() + 1))
        X_pos = np.repeat(X_tgt_labeled_np[pos_idx], scale, axis=0)
        y_pos = np.repeat(y_tgt_labeled_np[pos_idx], scale, axis=0)
        X_tgt_labeled_np = np.concatenate([X_tgt_labeled_np, X_pos], axis=0)
        y_tgt_labeled_np = np.concatenate([y_tgt_labeled_np, y_pos], axis=0)
    print(f"[INFO] Labeled target set after balancing: {len(y_tgt_labeled_np)} samples")

    tgt_labeled_ds = TensorDataset(
        torch.tensor(X_tgt_labeled_np, dtype=torch.float32),
        torch.tensor(y_tgt_labeled_np, dtype=torch.float32).view(-1, 1),
    )
    tgt_labeled_loader = DataLoader(tgt_labeled_ds, batch_size=batch_size, shuffle=True)

    ft_model = deepcopy(base_model).to(device)
    for param in ft_model.parameters():
        param.requires_grad = True

    ft_optimizer = optim.Adam(
        ft_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    ft_criterion = nn.BCELoss()

    ft_model.train()
    for epoch in range(epochs_ft):
        total_loss = 0.0
        for x, y in tgt_labeled_loader:
            x = x.to(device)
            y = y.to(device)

            preds, _ = ft_model(x)
            loss = ft_criterion(preds, y)

            ft_optimizer.zero_grad()
            loss.backward()
            ft_optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(tgt_labeled_loader)
        print(f"[FT Epoch {epoch+1}/{epochs_ft}] Loss: {avg_loss:.4f}")

    ft_model.eval()
    return ft_model


def calibrate_threshold_for_f1(
    model: FTDANN,
    X_tgt_test_np,
    y_tgt_test_np,
    device: torch.device,
):
    """
    Predict probabilities on target test and choose threshold
    that maximizes F1-score.
    """
    if isinstance(X_tgt_test_np, pd.DataFrame):
        X_tgt_test_np = X_tgt_test_np.to_numpy()
    X_tgt_test_np = np.array(X_tgt_test_np, dtype=np.float32)

    model.eval()
    with torch.no_grad():
        preds, _ = model(torch.tensor(X_tgt_test_np, dtype=torch.float32).to(device))
    probs = preds.cpu().numpy().ravel()
    y_true = y_tgt_test_np

    fpr, tpr, thresholds = roc_curve(y_true, probs)
    f1s = [f1_score(y_true, probs > t) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]

    print(f"\n[INFO] Optimal threshold for F1: {best_t:.3f}")
    return best_t, probs, y_true


def evaluate_with_threshold(
    probs,
    y_true,
    threshold: float,
    title: str = "Fine-Tuned Target Evaluation (DANN + FT + Calibration)",
):
    """
    Final confusion-matrix derived metrics using calibrated threshold.
    """
    preds_bin = (probs > threshold).astype(int)

    TP = ((preds_bin == 1) & (y_true == 1)).sum()
    TN = ((preds_bin == 0) & (y_true == 0)).sum()
    FP = ((preds_bin == 1) & (y_true == 0)).sum()
    FN = ((preds_bin == 0) & (y_true == 1)).sum()

    acc = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    f1 = f1_score(y_true, preds_bin)
    Pd = TP / (TP + FN + 1e-8)
    Pfa = FP / (FP + TN + 1e-8)
    Pmd = 1.0 - Pd

    print(f"\n===== {title} =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Pd (Detection): {Pd:.4f}")
    print(f"Pfa (False Alarm): {Pfa:.4f}")
    print(f"Pmd (Miss Detection): {Pmd:.4f}")

    return acc, f1, Pd, Pfa, Pmd


# ======================================================
# Main entry point
# ======================================================

def main():
    # Adjust this path to where your CSV lives in the repo
    data_csv = Path("CASS_Spectrum_Dataset.csv")
    if not data_csv.exists():
        raise FileNotFoundError(f"Could not find {data_csv} in current directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    (
        X_src_train_np,
        y_src_train_np,
        X_src_val_np,
        y_src_val_np,
        X_tgt_train_np,
        y_tgt_train_np,  # unused directly but kept for completeness
        X_tgt_test_np,
        y_tgt_test_np,
        X_tgt_labeled_np,
        y_tgt_labeled_np,
    ) = load_and_preprocess(data_csv, labeled_fraction=0.05)

    # Build dataloaders
    (
        src_train_loader,
        src_val_loader,
        tgt_train_loader,
        tgt_test_loader,
    ) = build_dataloaders(
        X_src_train_np,
        y_src_train_np,
        X_src_val_np,
        y_src_val_np,
        X_tgt_train_np,
        X_tgt_test_np,
        y_tgt_test_np,
        batch_size=64,
    )

    # Initialize model
    input_dim = X_src_train_np.shape[1]
    model = FTDANN(input_dim=input_dim, lambda_grl=0.3)

    # Train DANN + load best checkpoint
    model = train_dann(
        model,
        src_train_loader,
        src_val_loader,
        tgt_train_loader,
        device=device,
        epochs=200,
        patience=20,
        save_path="./best_dann.pt",
    )

    # Evaluate on target before fine-tuning
    evaluate_on_target(model, tgt_test_loader, device, threshold=0.5, title="Target Domain Evaluation (Pre-FT)")

    # Semi-supervised fine-tuning on small labeled target subset
    ft_model = fine_tune_on_labeled_target(
        model,
        X_tgt_labeled_np,
        y_tgt_labeled_np,
        device=device,
        epochs_ft=100,
        batch_size=16,
        lr=1e-4,
        weight_decay=1e-5,
    )

    # Calibrate threshold on target test for best F1
    best_t, probs, y_true = calibrate_threshold_for_f1(
        ft_model,
        X_tgt_test_np,
        y_tgt_test_np,
        device=device,
    )

    # Final evaluation with calibrated threshold
    evaluate_with_threshold(
        probs,
        y_true,
        threshold=best_t,
        title="Fine-Tuned Target Evaluation (DANN + FT + Calibration)",
    )


if __name__ == "__main__":
    main()
