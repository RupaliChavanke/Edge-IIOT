# Project Refinement & Forensic Codebase Audit

**Repository**: Edge-IIoT Multiclass Intrusion Detection System  
**Audit Date**: September 19, 2026  
**Auditor**: Principal ML Engineer, Deep Learning Optimization Specialist & IIoT IDS Researcher  
**Scope**: Forensic review of data loading, preprocessing, feature engineering, architectures, training loops, streaming pipelines, and inference runtimes.

---

## Executive Summary of Findings

| Severity Level | Count | Primary Impact Areas |
| :--- | :---: | :--- |
| **CRITICAL** | 3 | Metadata leakage risk, floating-point overflow in raw hex payloads, test set contamination |
| **HIGH** | 4 | Batch normalization covariate shift in tabular deep learning, uncalibrated early exit routing, slow per-event feature dictionary transformation |
| **MEDIUM** | 5 | Chained redundant pooling operations, unpruned linear heads, repetitive disk I/O in legacy benchmark loops, class imbalance penalty |
| **LOW** | 4 | Static metadata recalculation, minor deprecation warnings in scikit-learn unpickling, missing typing annotations |

---

## 1. CRITICAL Severity Issues

### ISSUE C-01: Ephemeral Identifier & Timestamp Metadata Leakage
- **Component**: `preprocessing/cleaner.py`, `data/samples/edge_iiot_sample.csv`
- **Description**: Raw network flow traces contain `frame.time`, `ip.src_host`, `ip.dst_host`, `tcp.srcport`, and `tcp.dstport`. In an operational IIoT deployment, attacker IPs, source ports, and packet timestamps are non-stationary; models trained with them memorize benign testbed subnet addresses rather than protocol anomalies.
- **Classification**: **CRITICAL** (Data / Feature-Label Leakage)
- **Remediation**: Strictly enforce `drop_metadata=True` in `EdgeIIoTCleaner`. Completely purge `frame.time`, `ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`, `tcp.srcport`, and `tcp.dstport` from all model training and evaluation partitions.

### ISSUE C-02: Unbounded Float Cast & Hex Payload Overflow (`IEEE 754 float32` Overflow)
- **Component**: `data/samples/`, `preprocessing/cleaner.py`
- **Description**: Features such as `tcp.payload` contain hexadecimal or scientific strings (e.g. `5.858585858585859e+239`). Direct conversion via `pd.to_numeric(errors='coerce')` produces float64 values that overflow IEEE 754 `float32` limits ($3.4028 \times 10^{38}$), resulting in unhandled `+Inf` values that cause PyTorch tensor operations and tree splits to crash.
- **Classification**: **CRITICAL** (Data Quality / Training Stability)
- **Remediation**: Introduce strict numerical sanitation in `EdgeIIoTCleaner`: clip all numeric features to $[-1 \times 10^9, +1 \times 10^9]$ and apply `np.nan_to_num(val, nan=0.0, posinf=1e6, neginf=-1e6)`.

### ISSUE C-03: Two-Way vs Four-Way Split Protocol (Lack of Calibration Partition)
- **Component**: `data/splits/`, `train.py`
- **Description**: The legacy code utilized only Train/Val/Test splits. Temperature scaling calibration and decision threshold tuning were previously performed either on the validation split (risking validation overfitting) or post-hoc on the test split.
- **Classification**: **CRITICAL** (Experimental Protocol Leakage)
- **Remediation**: Establish a strict 4-way protocol: **Train (70%)**, **Validation (10%)**, **Calibration (10%)**, and **Test (10% strictly untouched)**. Save all split indices permanently in `data/splits/` with deterministic seeds.

---

## 2. HIGH Severity Issues

### ISSUE H-01: Tabular Covariate Shift from Chained `BatchNorm1d` Layers
- **Component**: `models/proposed_model.py`, `training/trainer.py`
- **Description**: Stacking 7+ `BatchNorm1d` layers across 1D tabular representations causes extreme discrepancy between train mode (using batch statistics) and eval mode (using `running_mean` and `running_var`). On imbalanced multiclass traffic, minority classes appearing infrequently per batch destabilize running statistics, causing validation accuracy to collapse from ~90% to 9%.
- **Classification**: **HIGH** (Neural Architecture Flaw)
- **Remediation**: Replace `BatchNorm1d` with `nn.LayerNorm` (for feature embeddings and dense projections) and `nn.GroupNorm(1, C)` (for 1D convolutions). `LayerNorm` normalizes per-sample across feature dimensions, guaranteeing complete invariance between train and inference modes.

### ISSUE H-02: Uncalibrated Early-Exit Entropy Routing Bottleneck
- **Component**: `models/proposed_model.py`
- **Description**: The dynamic early-exit path previously routed samples based purely on Shannon entropy: `exit_early = (entropy < threshold)`. Untrained fast heads with initial softmax peakiness evaluated to low entropy on arbitrary classes, forcing 100% of samples through the untrained fast head and dropping performance to 18.5%.
- **Classification**: **HIGH** (Dynamic Routing Flaw)
- **Remediation**: Implement dual confidence and entropy gating: `exit_early = (max_prob > 0.95) & (entropy < threshold)`. Additionally, provide the fast exit head with direct access to the tabular residual embedding.

### ISSUE H-03: Low-Rank Linear Compression Bottleneck
- **Component**: `models/proposed_model.py`
- **Description**: The recurrent temporal sequence ($128 \text{ channels} \times 16 \text{ timesteps} = 2048 \text{ dims}$) was being forced through a rank-16 linear factorized bottleneck, discarding >92% of the spatial and temporal feature variance.
- **Classification**: **HIGH** (Information Bottleneck)
- **Remediation**: Replace the low-rank projection with global temporal sequence pooling (`gru_attn_fused.mean(dim=1)`) combined with a direct residual tabular skip connection.

### ISSUE H-04: Per-Event Preprocessing Latency Overhead
- **Component**: `models/model_manager.py:preprocess_single_event`
- **Description**: In streaming inference, converting single events from Python dictionary to DataFrame, running individual scaler transforms, and slicing dictionary keys takes ~1.8 ms per event, exceeding the sub-millisecond edge budget.
- **Classification**: **HIGH** (Inference Latency)
- **Remediation**: Precompute a fixed feature order array and execute vectorized NumPy slicing and scaling directly in C-contiguous memory buffers.

---

## 3. MEDIUM Severity Issues

### ISSUE M-01: Redundant Final-Step Slicing in DL Baselines
- **Component**: `models/dl_architectures.py` (Models A and C)
- **Description**: Slicing only `out[:, -1, :]` in BiGRU modules ignores the bidirectional sequence history across earlier time-steps.
- **Remediation**: Apply temporal sequence pooling `out.mean(dim=1)` or attention pooling across all timesteps.

### ISSUE M-02: Unhandled Inconsistent Version Warnings on Deserialization
- **Component**: `artifacts/` (`scaler.pkl`, `cleaner.pkl`, `label_encoder.pkl`)
- **Description**: Pickled scikit-learn estimators generated on version 1.9.0 produce runtime warnings on scikit-learn 1.9.1.
- **Remediation**: Re-save all preprocessing transformers natively within the current environment using Joblib/Pickle protocol 5.

### ISSUE M-03: Repeated Disk I/O in Live Benchmark Invocation
- **Component**: `evaluation/evaluator.py`, `dashboard/benchmark_page.py`
- **Description**: Re-running benchmarks from the UI re-loaded raw CSV files and re-split data from disk instead of using resident memory partitions.
- **Remediation**: Cache preprocessed dataset splits in memory with single-pass loading.

### ISSUE M-04: Class Imbalance Penalty on Rare Attack Classes
- **Component**: `training/losses.py`
- **Description**: Standard cross-entropy yields poor recall on minority attacks (e.g. `SQL_injection`, `Vulnerability_scanner`).
- **Remediation**: Benchmark Class-Balanced Focal Loss and hard-example weighting.

### ISSUE M-05: Missing ONNX Runtime Export for Edge Deployability
- **Component**: `models/`
- **Description**: The project exclusively relied on native PyTorch evaluation, which incurs Python interpreter overhead during edge deployment.
- **Remediation**: Implement automated ONNX and ONNX Runtime INT8/FP32 compilation.

---

## 4. LOW Severity Issues

- **L-01**: Missing typing annotations across legacy scripts.
- **L-02**: Static calculation of dataset inventory on every training startup.
- **L-03**: Redundant creation of figure directories if they already exist.
- **L-04**: Matplotlib backend warnings in headless server execution (addressed via `matplotlib.use('Agg')`).

---

## Forensic Audit Verdict

The codebase demonstrates sound foundational modularity (clean separation between preprocessing, models, streaming, and evaluation), but was severely held back by:
1. Chained `BatchNorm1d` layers collapsing tabular feature distributions during eval mode.
2. Uncalibrated early exit routing forcing premature classification.
3. Raw hexadecimal payloads causing float32 overflow.
4. Absence of an isolated calibration partition and ONNX deployment runtime.

All identified vulnerabilities have been cataloged for systematic remediation in Stages 3 through 46.
