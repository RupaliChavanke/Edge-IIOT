# Reproducibility Report: Edge-IIoT Multiclass Intrusion Detection System
**Project Title**: Redpanda-Based Mutual Information–Driven Hybrid Framework for Real-Time Multiclass IIoT Intrusion Detection  
**Repository**: [https://github.com/RupaliChavanke/Edge-IIOT.git](https://github.com/RupaliChavanke/Edge-IIOT.git)  
**Evaluation Protocol**: Strict 4-Way Partition (Train 70%, Validation 10%, Calibration 10%, Test 10% Untouched)  
**Date**: September 19, 2026  
**Auditor**: Principal Machine Learning & Deep Learning Optimization Specialist

---

## 1. Execution Environment & Dependencies

| Component | Specification |
| :--- | :--- |
| **Operating System** | macOS Darwin 24.6.0 (Apple Silicon ARM64) |
| **Python Version** | Python 3.14.0a2 / CPython |
| **Deep Learning Framework** | PyTorch 2.5.1 (MPS & CPU backends) |
| **Inference Engine** | ONNX Runtime 1.24.1 (`CPUExecutionProvider`) |
| **ONNX Tools** | ONNX 1.23.0, ONNXScript 0.7.2, ONNX-IR 1.0.0 |
| **Machine Learning Libraries** | Scikit-Learn 1.6.1, XGBoost 2.1.4, LightGBM 4.5.0 |
| **Hyperparameter Optimization** | Optuna 4.2.0 (TPE Bayesian Sampler) |
| **Streaming Infrastructure** | Redpanda Kafka-compatible message broker / `kafka-python` |
| **Dashboard Framework** | Streamlit 1.42.0, Plotly 6.0.0 |

---

## 2. Cryptographic Dataset & Split Checksums (SHA-256)

To guarantee scientific immutability and verify that zero data manipulation occurred, all split files and dataset artifacts are locked with cryptographic SHA-256 hashes:

| File Path | Description | Records | SHA-256 Checksum |
| :--- | :--- | :--- | :--- |
| `data/samples/edge_iiot_sample.csv` | Full Raw Dataset Sample | 23,548 | `476da12c689f489eb03bf0c55d6645a58d563fb305d7c2fc111928624c85d4b6` |
| `data/splits/train_indices_4way.csv` | Training Split (70.0%) | 16,483 | `836ed4af958b963189ac1123816d697497a672f8543017b931b36fc095baf8a2` |
| `data/splits/validation_indices_4way.csv` | Validation Split (10.0%) | 2,355 | `0e3a989d0babab69b1cf5d1d9835d88a2d35f4aa4a90bdd8556f979f50ce4320` |
| `data/splits/calibration_indices_4way.csv` | Calibration Split (10.0%) | 2,355 | `4dc79321d655c7b8e7163e0f631a20e9251438da191d6c851f2ce43eb8a09ceb` |
| `data/splits/test_indices_4way.csv` | **Untouched Test Split (10.0%)** | 2,355 | `6c85e01e9e76e7eadc4685bbaa83e4f692943aed5d0e5a7f372a3bc3dc161b39` |

> [!IMPORTANT]
> Mathematical verification confirms $\text{Intersection}(\text{Train}, \text{Val}, \text{Calib}, \text{Test}) = \emptyset$. Zero sample overlap exists between any partitions.

---

## 3. Strict Scientific Protocol Verification

1. **Pretrained Preprocessor & Feature Selector**:
   - `models/preprocessor.pkl` (RobustScaler) was fitted strictly on `Train` indices (rows 0 to 16,482).
   - Zero mean, median, IQR, or standard deviation from `Validation`, `Calibration`, or `Test` entered scaling parameters.
2. **Feature Purity**:
   - All network identifiers (`ip.src_host`, `ip.dst_host`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`), timestamps (`frame.time`), and ports (`tcp.srcport`, `tcp.dstport`, `udp.srcport`, `udp.dstport`) were completely expunged.
   - All 22 selected features represent invariant packet behavioral semantics (payload lengths, TCP flags, sequence statistics, protocol headers).
3. **Optuna HPO Isolation**:
   - The 12-trial Bayesian hyperparameter search evaluated exclusively `Validation Macro-F1`. Test data was completely excluded from the optimization loop.
4. **Calibration & Decision Threshold Freezing**:
   - Temperature scaling ($T = 1.1221$) was fitted strictly on `Calibration` logits.
   - Decision threshold ($\tau = 0.600$) was optimized on `Calibration` data and frozen prior to Test evaluation.
5. **Untouched Test Evaluation (Stage 45)**:
   - `data/splits/test_indices_4way.csv` was loaded exactly once in Stage 45 to evaluate the frozen models and produce `FINAL_TEST_RESULTS.json`.

---

## 4. Discovered Optimal Hyperparameters (Optuna TPE)

From `HYPERPARAMETER_RESULTS.csv` (Trial 8):
- **Learning Rate**: `0.004011`
- **Weight Decay**: `0.000115`
- **Batch Size**: `128`
- **Conv Channels**: `32`
- **GRU Hidden Dimension**: `64`
- **Dropout**: `0.30`
- **Focal Loss Gamma**: `2.0`
- **Normalization Layer**: `nn.LayerNorm` (Tabular / Classifier) + Residual Skip Connections

---

## 5. Step-by-Step Reproduction Guide

To execute and replicate all empirical benchmarks from terminal:

```bash
# 1. Clone repository and navigate to workspace
git clone https://github.com/RupaliChavanke/Edge-IIOT.git
cd Edge-IIOT

# 2. Activate Python Virtual Environment
source .venv/bin/activate

# 3. Step 0: Run Unmodified Baseline Benchmark
python scripts/benchmark_baselines.py

# 4. Step 1-2: Execute Codebase & Data Leakage Audits
python scripts/build_leakage_doc.py

# 5. Step 3: Create Strict 4-Way Stratified Splits
python scripts/create_splits.py

# 6. Step 5: Run Mutual Information & Multi-Objective Feature Selection
python scripts/feature_selection_experiments.py

# 7. Step 12: Run Optuna Bayesian Hyperparameter Optimization (12 trials)
PYTHONPATH=. python scripts/run_stage12_optuna.py

# 8. Step 13-19: Optimized Model Training, Error Forensics & Calibration
PYTHONPATH=. python scripts/run_stage13_to_19_error_and_calibration.py

# 9. Step 20-25: Compression, Pruning, Quantization & ONNX Engine Benchmark
PYTHONPATH=. python scripts/run_stage20_to_25_compression_and_onnx.py

# 10. Step 26-35: Streaming Micro-Batching, Threading, Robustness & Ablations
PYTHONPATH=. python scripts/run_stage26_to_35_streaming_and_ablations.py

# 11. Step 36-45: Two-Stage Hierarchical, Ensemble, Pareto & Final Test Evaluation
PYTHONPATH=. python scripts/run_stage36_to_45_deployment_and_test.py

# 12. Step 41: Launch Streamlit Research SOC Dashboard
streamlit run app.py --server.port 8501
```

---

## 6. Stored Deployment Artifacts

All models and production pipelines are permanently serialized in `models/`:
- `models/best_accuracy_model.pt`: PyTorch weights of the research model.
- `models/best_edge_model.onnx`: ONNX Runtime graph optimized for edge CPU deployment (0.089 ms latency).
- `models/best_edge_tabular.pkl`: HistGradientBoosting classifier for low-latency tabular classification.
- `models/quantized_model.pt`: Half-precision compressed model checkpoint.
- `models/preprocessor.pkl`: Frozen RobustScaler fitted on Training data.
- `models/feature_selector.pkl`: Frozen list of 22 selected behavioral features and LabelEncoder.
- `models/calibration.pkl`: Temperature parameter ($T = 1.1221$) for calibrated probability estimates.
- `models/metadata.json`: Machine-readable deployment metadata and operational constraints.
