"""
Leakage-Free Splitting Engine for Edge-IIoTset.
Generates and permanently persists index files for:
- Protocol A: Stratified Random Split (70% Train, 15% Val, 15% Test) for seeds [42, 52, 62, 72, 82]
- Protocol B: Temporal Block Split (chronological ordering by frame.time)
- Protocol C: Device / IP-aware Split (holding out virtual machines/IPs)
- Protocol E: 5-Fold Stratified Cross-Validation folds
Outputs stored in: data/splits/
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def create_and_save_splits(csv_path="data/samples/edge_iiot_sample.csv", splits_dir="data/splits"):
    os.makedirs(splits_dir, exist_ok=True)
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    n_samples = len(df)
    target = df["Attack_type"].astype(str)
    indices = np.arange(n_samples)

    # 1. Protocol A: Multi-seed Stratified Splits (42, 52, 62, 72, 82)
    seeds = [42, 52, 62, 72, 82]
    for seed in seeds:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.30, stratify=target, random_state=seed
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, stratify=target.iloc[temp_idx], random_state=seed
        )

        pd.DataFrame({"index": train_idx}).to_csv(os.path.join(splits_dir, f"train_indices_seed{seed}.csv"), index=False)
        pd.DataFrame({"index": val_idx}).to_csv(os.path.join(splits_dir, f"val_indices_seed{seed}.csv"), index=False)
        pd.DataFrame({"index": test_idx}).to_csv(os.path.join(splits_dir, f"test_indices_seed{seed}.csv"), index=False)

        if seed == 42:
            # Default canonical benchmark splits
            pd.DataFrame({"index": train_idx}).to_csv(os.path.join(splits_dir, "train_indices.csv"), index=False)
            pd.DataFrame({"index": val_idx}).to_csv(os.path.join(splits_dir, "validation_indices.csv"), index=False)
            pd.DataFrame({"index": test_idx}).to_csv(os.path.join(splits_dir, "test_indices.csv"), index=False)
            print(f"Canonical Seed 42 Split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # 2. Protocol B: Temporal Chronological Split (70% Train, 15% Val, 15% Test)
    if "frame.time" in df.columns:
        # Sort by timestamp
        time_sorted_idx = df["frame.time"].argsort().values
        t_tr_end = int(0.70 * n_samples)
        t_val_end = int(0.85 * n_samples)
        temp_tr = time_sorted_idx[:t_tr_end]
        temp_val = time_sorted_idx[t_tr_end:t_val_end]
        temp_te = time_sorted_idx[t_val_end:]
        pd.DataFrame({"index": temp_tr}).to_csv(os.path.join(splits_dir, "temporal_train_indices.csv"), index=False)
        pd.DataFrame({"index": temp_val}).to_csv(os.path.join(splits_dir, "temporal_val_indices.csv"), index=False)
        pd.DataFrame({"index": temp_te}).to_csv(os.path.join(splits_dir, "temporal_test_indices.csv"), index=False)
        print(f"Protocol B Temporal Split: Train={len(temp_tr)}, Val={len(temp_val)}, Test={len(temp_te)}")

    # 3. Protocol C: Device/IP-Aware Split (Holding out entire IPs)
    if "ip.src_host" in df.columns:
        unique_ips = df["ip.src_host"].unique()
        np.random.seed(42)
        shuffled_ips = np.random.permutation(unique_ips)
        n_ips = len(shuffled_ips)
        tr_ips = set(shuffled_ips[:int(0.70 * n_ips)])
        val_ips = set(shuffled_ips[int(0.70 * n_ips):int(0.85 * n_ips)])
        te_ips = set(shuffled_ips[int(0.85 * n_ips):])

        dev_tr = indices[df["ip.src_host"].isin(tr_ips)]
        dev_val = indices[df["ip.src_host"].isin(val_ips)]
        dev_te = indices[df["ip.src_host"].isin(te_ips)]
        pd.DataFrame({"index": dev_tr}).to_csv(os.path.join(splits_dir, "device_train_indices.csv"), index=False)
        pd.DataFrame({"index": dev_val}).to_csv(os.path.join(splits_dir, "device_val_indices.csv"), index=False)
        pd.DataFrame({"index": dev_te}).to_csv(os.path.join(splits_dir, "device_test_indices.csv"), index=False)
        print(f"Protocol C Device-Aware Split: Train={len(dev_tr)}, Val={len(dev_val)}, Test={len(dev_te)}")

    # 4. Protocol E: 5-Fold Stratified Cross-Validation Indices
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_k, te_k) in enumerate(skf.split(indices, target)):
        pd.DataFrame({"index": tr_k}).to_csv(os.path.join(splits_dir, f"cv_fold{fold}_train_indices.csv"), index=False)
        pd.DataFrame({"index": te_k}).to_csv(os.path.join(splits_dir, f"cv_fold{fold}_test_indices.csv"), index=False)
    print("Protocol E: Saved 5-Fold Stratified Cross-Validation index sets.")

if __name__ == "__main__":
    create_and_save_splits()
