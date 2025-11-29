# data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def prepare_cass_data(
    file_path: str,
    labeled_fraction: float = 0.05,
    seed: int = 42,
):
    """
    Load and preprocess the CASS spectrum dataset.

    Returns:
        A dict of numpy arrays:
        {
            'X_src_train', 'y_src_train',
            'X_src_val',   'y_src_val',
            'X_tgt_train', 'y_tgt_train',
            'X_tgt_test',  'y_tgt_test',
            'X_tgt_labeled', 'y_tgt_labeled',
            'input_dim'
        }
    """
    data = pd.read_csv(file_path)

    # Basic cleaning
    data = data.dropna().copy()
    data["Target"] = data["Target"].astype(int)

    # Base features and label
    base_features = [
        "PU_Signal_Strength",
        "Frequency_Band",
        "SNR_SU1", "SNR_SU2", "SNR_SU3",
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

    # Source = high SNR clusters, Target = low SNR clusters
    source_clusters = sorted_clusters.index[: num_clusters // 2]
    target_clusters = sorted_clusters.index[num_clusters // 2 :]

    # --- Split domains ---
    X_source_raw = data[data["Cluster_ID"].isin(source_clusters)][base_features]
    y_source = data[data["Cluster_ID"].isin(source_clusters)][label_col]

    X_target_raw = data[data["Cluster_ID"].isin(target_clusters)][base_features]
    y_target = data[data["Cluster_ID"].isin(target_clusters)][label_col]

    # --- Train/Val/Test splits ---
    X_src_train, X_src_val, y_src_train, y_src_val = train_test_split(
        X_source_raw,
        y_source,
        test_size=0.3,
        stratify=y_source,
        random_state=seed,
    )

    X_tgt_train_unlabeled, X_tgt_test, y_tgt_train_unlabeled, y_tgt_test = train_test_split(
        X_target_raw,
        y_target,
        test_size=0.3,
        stratify=y_target,
        random_state=seed,
    )

    # Labeled subset from target (small fraction)
    num_labeled = int(len(X_tgt_train_unlabeled) * labeled_fraction)
    X_tgt_labeled = X_tgt_train_unlabeled.sample(num_labeled, random_state=seed)
    y_tgt_labeled = y_tgt_train_unlabeled.loc[X_tgt_labeled.index]

    # --- One-hot encode Frequency_Band ---
    def encode(df):
        return pd.get_dummies(df, columns=["Frequency_Band"])

    X_src_train_enc = encode(X_src_train)
    X_src_val_enc = encode(X_src_val)
    X_tgt_train_enc = encode(X_tgt_train_unlabeled)
    X_tgt_test_enc = encode(X_tgt_test)
    X_tgt_labeled_enc = encode(X_tgt_labeled)

    # Align columns
    all_cols = sorted(
        set(X_src_train_enc.columns)
        | set(X_src_val_enc.columns)
        | set(X_tgt_train_enc.columns)
        | set(X_tgt_test_enc.columns)
        | set(X_tgt_labeled_enc.columns)
    )
    X_src_train_enc = X_src_train_enc.reindex(columns=all_cols, fill_value=0)
    X_src_val_enc = X_src_val_enc.reindex(columns=all_cols, fill_value=0)
    X_tgt_train_enc = X_tgt_train_enc.reindex(columns=all_cols, fill_value=0)
    X_tgt_test_enc = X_tgt_test_enc.reindex(columns=all_cols, fill_value=0)
    X_tgt_labeled_enc = X_tgt_labeled_enc.reindex(columns=all_cols, fill_value=0)

    # --- Standardize continuous features (fit on source train only) ---
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

    # --- Convert to numpy ---
    X_src_train_np = X_src_train_enc.to_numpy(dtype=np.float32)
    y_src_train_np = y_src_train.to_numpy(dtype=np.float32)

    X_src_val_np = X_src_val_enc.to_numpy(dtype=np.float32)
    y_src_val_np = y_src_val.to_numpy(dtype=np.float32)

    X_tgt_train_np = X_tgt_train_enc.to_numpy(dtype=np.float32)
    y_tgt_train_np = y_tgt_train_unlabeled.to_numpy(dtype=np.float32)

    X_tgt_test_np = X_tgt_test_enc.to_numpy(dtype=np.float32)
    y_tgt_test_np = y_tgt_test.to_numpy(dtype=np.float32)

    X_tgt_labeled_np = X_tgt_labeled_enc.to_numpy(dtype=np.float32)
    y_tgt_labeled_np = y_tgt_labeled.to_numpy(dtype=np.float32)

    return {
        "X_src_train": X_src_train_np,
        "y_src_train": y_src_train_np,
        "X_src_val": X_src_val_np,
        "y_src_val": y_src_val_np,
        "X_tgt_train": X_tgt_train_np,
        "y_tgt_train": y_tgt_train_np,
        "X_tgt_test": X_tgt_test_np,
        "y_tgt_test": y_tgt_test_np,
        "X_tgt_labeled": X_tgt_labeled_np,
        "y_tgt_labeled": y_tgt_labeled_np,
        "input_dim": X_src_train_np.shape[1],
    }
