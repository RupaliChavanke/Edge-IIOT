# Comprehensive Repository Audit Report: Edge-IIoTset IDS Framework

**Date**: September 2026  
**Repository**: `https://github.com/RupaliChavanke/Edge-IIOT`  
**Project**: Redpanda-Based Mutual Information–Driven Hybrid CNN–BiGRU Framework for Real-Time Multiclass IIoT Intrusion Detection  
**Author / Investigator**: Senior Machine Learning & Cybersecurity Research Team  

---

## Executive Summary

This forensic and technical audit inspects the entirety of the existing Edge-IIoT codebase, including `app.py`, `train.py`, `config.yaml`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `preprocessing/`, `training/`, `models/`, `checkpoints/`, `evaluation/`, `experiments/`, `dashboard/`, `streaming/`, `visualization/`, `tests/`, `data/`, and `artifacts/`.

The primary goal of the project is to produce an edge-deployable, leakage-free, computationally efficient, high-performance intrusion detection system (IDS) targeting $\ge 98\%$ across key multiclass metrics (Accuracy, Macro-Precision, Macro-Recall, Macro-F1, ROC-AUC, PR-AUC), while rigorously exposing and controlling data leakage.

Below is the exhaustive 25-point audit of the existing repository state prior to upgrade.

---

## 1. Current Architecture
The system consists of three primary layers:
1. **Streaming Backbone**: Native Redpanda C++ broker running in Docker (exposing Kafka wire protocol at `localhost:19092`) with fallback to a Python thread-safe `InMemoryStreamingBus` for local / cloud / zero-broker execution.
2. **Deep Learning Core**: Hybrid neural network (`ProposedHybridEdgeIIoTModel`) combining 1D Depthwise Separable Convolutions, a 1D Ghost Module, Squeeze-and-Excitation (SE) attention, an entropy-gated early exit, a bidirectional GRU, multi-head temporal self-attention, low-rank linear projection, and a tabular embedding skip connection.
3. **Operations & Visualization**: Streamlit multi-page dashboard (`app.py`), offline model manager (`ModelManager`), and batch/streaming inference pipelines.

## 2. Current Data Pipeline
- Data is ingested from `data/samples/edge_iiot_sample.csv` (23,548 rows, 63 columns) or `data/raw/ML-EdgeIIoT-dataset.csv` via `preprocessing.loader.EdgeIIoTDataLoader`.
- Missing target rows are dropped. Exact duplicates are pruned.
- The pipeline splits data into Train (70%), Validation (15%), and Test (15%) using `train_test_split(stratify=y)`.

## 3. Current Preprocessing Pipeline
- `EdgeIIoTCleaner`:
  - Drops 14 metadata/identifier columns (`frame.time`, `ip.src_host`, `ip.dst_host`, `arp.*`, `http.file_data`, `http.request.full_uri`, `http.referer`, `http.request.uri.query`, `tcp.payload`, `tcp.options`, `mqtt.msg`, `dns.qry.name*`).
  - Imputes numerical missing values with median and categorical with mode.
  - Hand-extracts 5 regex indicators: `sig_sql`, `sig_xss`, `sig_upload`, `sig_password`, `sig_cve`, plus 5 length attributes (`len_uri`, `len_query`, `len_payload`, `len_referer`, `len_file_data`).
- `EdgeIIoTEncoder`: Fits `LabelEncoder` for target attack classes and categorical features.
- `EdgeIIoTScaler`: Applies `RobustScaler` (median / IQR) with optional outlier clamping ($\pm 5.0 \times \text{IQR}$).

## 4. Current Feature-Selection Pipeline
- Implemented in `preprocessing.mrmr_jmi.py` (`MRMRJMISelector`):
  - Calculates mutual information ($I(X_i; Y)$) using `mutual_info_classif(n_neighbors=3)` on a subsample of 15,000 training points.
  - Greedy Joint Mutual Information (JMI) iterative selection: maximizes $I(X_i; Y)$ while penalizing average Pearson correlation with already selected features.
  - Target dimension is set by `config.yaml` (`mrmr_k_features: 22`).

## 5. Current Model Architecture
- **Input Dimension**: $k=22$ features (or padded to 16/32).
- **Tabular Skip Pathway**: Two-layer MLP (`Linear(22, 256) -> BN -> GELU -> Dropout -> Linear(256, 128) -> BN -> GELU`).
- **Spatial Feature Extractor**: `DepthwiseSeparableConv1d(1 -> 32, k=3)` followed by `GhostModule1d(32 -> 64, ratio=2)` and `AdaptiveAvgPool1d(16)`.
- **Channel Attention**: `SEAttention1d(channels=64, reduction=8)`.
- **Fast Head (Early Exit)**: Flattened spatial features (1024) $\to$ `Linear(1024, 128) -> GELU -> Linear(128, 15)`.
- **Temporal Deep Path**: Transposed features $(B, 16, 64) \to$ `TemporalBiGRU(64 -> 128)` $\to$ `MultiHeadTemporalAttention(128, heads=4)`.
- **Dimensionality Reduction**: `LowRankLinear(2048 -> 128, rank=16)`.
- **Final Classification Head**: Concatenation of deep latent representation (128) + tabular skip embedding (128) = 256 $\to$ `Linear(256, 128) -> BN -> GELU -> Dropout -> Linear(128, 15)`.

## 6. Current Loss Functions
- Implemented in `training.losses.py`:
  - `FocalLoss`: Multiclass focal loss ($\gamma = 2.0$) with target clamping ($10^{-8}, 1 - 10^{-8}$).
  - `CenterLoss`: Learns 15 class center vectors in 128-D latent space to minimize intra-class Euclidean variance.
  - `ProposedCompoundLoss`:
    $$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{focal}}(\hat{p}^{\text{fast}}, y) + 0.5 \cdot \mathcal{L}_{\text{focal}}(\hat{p}^{\text{deep}}, y) + 0.01 \cdot \mathcal{L}_{\text{center}}(z_{\text{latent}}, y)$$

## 7. Current Training Procedure
- Implemented in `training.trainer.py`:
  - Optimizer: `AdamW` ($\text{lr} = 0.001$, $\text{weight\_decay} = 10^{-4}$) optimizing model weights + center loss centers.
  - Scheduler: `ReduceLROnPlateau(factor=0.5, patience=2)`.
  - Gradient clipping: `max_norm = 2.0`.
  - Epochs: Configured to 10–15 epochs. Batch size = 64.

## 8. Current Validation Procedure
- Validation split is 15% of dataset.
- Evaluates dynamically using early exit entropy threshold $\tau = 0.35$.
- Tracks validation loss, validation accuracy, validation Macro-F1, and early exit percentage.

## 9. Current Test Procedure
- Evaluated on test partition with threshold $\tau = 0.35$.
- Computes comprehensive classification metrics via `training.metrics.calculate_comprehensive_metrics`.

## 10. Current Benchmark Models
- Scikit-learn baselines: Logistic Regression, Decision Tree, Random Forest, Extra Trees, MLP, Linear SVM.
- PyTorch deep learning baselines: 1D-CNN, LSTM, BiLSTM, GRU, BiGRU, CNN-LSTM, CNN-BiLSTM.

## 11. Current Reported Metrics
- In `artifacts/metrics.json`:
  - Accuracy: $95.66\%$
  - Macro Precision: $94.84\%$
  - Macro Recall: $94.59\%$
  - Macro F1: $94.68\%$
  - Weighted F1: $95.68\%$
  - ROC-AUC Macro: $99.85\%$
  - PR-AUC Macro: $97.01\%$
  - FPR: $0.0031$ ($0.31\%$)
  - FNR: $0.0541$ ($5.41\%$)

## 12. Current Computational Cost
- Total Parameters: 370,273 (~1.48 MB float32).
- MFLOPs: 0.741 MFLOPs per sample forward pass.
- Peak RAM: ~19.5 MB for isolated model execution.

## 13. Current Inference Latency
- Fast-path forward latency: ~0.052 ms.
- Deep-path forward latency: ~0.480 ms.
- Dynamic early exit latency (P50): ~0.024 ms / record under $\tau = 0.35$ batch mode.
- End-to-end Python streaming pipeline latency: ~5.00 ms / event.

---

## Critical Weaknesses & Vulnerability Analysis

### 14. Core Weaknesses
1. **Tabular-as-Sequence Limitation**: Reshaping 22 tabular features into a 1D sequence and applying 1D convolution assumes pseudo-spatial adjacency between arbitrary feature columns.
2. **Suboptimal Baseline Training in Benchmark Suite**: In `evaluation/evaluator.py`, deep baselines were trained for only 3 rapid epochs in a single batch, producing artificially depressed baseline numbers (15%–52%).
3. **Severe Confusions in Specific Classes**:
   - `DDoS_HTTP`: Recall 82.45%, F1 81.47% (278 misclassified as `Password`).
   - `Password` brute-force: Recall 77.85%, F1 79.25% (332 misclassified as `DDoS_HTTP`).
   - `Fingerprinting`: Recall 83.33% (only 150 total samples in dataset).

### 15. Potential Data Leakage
- `frame.time`: Timestamp ordering correlates with attack session execution periods in testbed generation.
- `ip.src_host` and `ip.dst_host`: Static attacker IP addresses (`192.168.0.128`, `192.168.0.170`) enable shortcut learning where models memorize host IPs rather than packet flow dynamics.

### 16. Potential Train/Test Contamination
- In existing `artifacts/test_samples.npz`, all 23,548 samples were stored as the test set rather than an isolated held-out split, contaminating the reported sample counts in Table 15 of the manuscript.

### 17. Potential Preprocessing Leakage
- While `fit_transform_pipeline` separates splits before fitting `EdgeIIoTCleaner` and `EdgeIIoTScaler`, hand-crafted regexes in `_extract_flow_features` (`sig_sql`, `sig_upload`) were derived from domain knowledge of specific attack strings present in the raw CSV text.

### 18. Potential Duplicate Leakage
- Exact duplicates in raw traffic flows must be removed prior to partitioning to prevent identical flow vectors from landing in both train and test splits.

### 19. Class Imbalance Issues
- The dataset is severely imbalanced:
  - `Normal`: 3,645 samples (15.48%)
  - `DDoS_UDP`: 2,175 samples (9.24%)
  - `MITM`: 60 samples (0.25%)
  - `Fingerprinting`: 150 samples (0.64%)
  - Rare classes suffer from high False Negative Rates without targeted loss weighting and center loss calibration.

### 20. Reproducibility Issues
- Lack of multi-seed cross-validation reporting (seed variation 42, 52, 62, 72, 82).
- Split indices were not stored to disk as persistent CSV files.

### 21. Code-Quality Problems
- Hardcoded parameter and FLOP multipliers in `evaluation/ablation.py` instead of actual retraining of ablated model variants.
- Missing `pydantic` and `confluent_kafka` in minimal environment scripts.
- Unhandled `CLOUD_MODE` return in `test_redpanda.py`.

### 22. Missing Experiments
- XGBoost, LightGBM / HistGradientBoosting, and CatBoost were absent from the official baseline runner.
- Ensembling (soft voting, stacking) was not evaluated.

### 23. Missing Ablation Studies
- Genuine component-level ablation requiring separate model architectures was not physically trained; only synthetic multipliers were logged.

### 24. Missing Statistical Validation
- No paired t-tests, Wilcoxon signed-rank tests, standard deviations, or 95% confidence intervals across repeated seed initializations.

### 25. Missing Security-Oriented Evaluation
- No adversarial feature perturbation testing, missing feature robustness evaluation, or class prior shift simulation.
- Lack of transparent severity classification rules in security alert publishing.

---

## Action Plan Summary

To achieve genuine research-grade standards, the upgrade will:
1. Enforce strict, leakage-free data partitioning with permanently persisted indices (`data/splits/`).
2. Separate the evaluation into **Experiment A (Conventional/Naive Protocol)** and **Experiment B (Leakage-Aware Protocol)**.
3. Replace synthetic ablation multipliers with genuinely trained and evaluated ablated models.
4. Expand baselines to include properly trained XGBoost, HistGradientBoosting, Random Forest, and full-epoch neural baselines.
5. Upgrade the proposed hybrid model with multi-scale depthwise separable convolutions, tabular embeddings, temperature calibration, and confidence threshold optimization.
6. Conduct 5-seed statistical validation, robustness testing under perturbation, and complete explainability via SHAP.
7. Refactor repository and upgrade Streamlit dashboard to all 24 required research pages.
