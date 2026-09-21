# Forensic Confusion Analysis: Edge-IIoTset Error Patterns

## 1. Executive Summary

- **Total Test Samples**: 3,533
- **Correct Predictions**: 3,483 (98.58%)
- **Total Misclassifications**: 50 (1.42%)
- **Benign Normal False Positives (Benign -> Attack)**: 2
- **Dangerous Missed Threats (Attack -> Normal)**: 0

---

## 2. Top Confused Attack Class Pairs

| Rank | True Attack Class | Misclassified As | Misclassification Count | Error Rate on True Class | Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Password` | `DDoS_HTTP` | 20 | 8.89% | Boundary probability distribution near decision hyperplane. |
| 2 | `DDoS_HTTP` | `Password` | 17 | 7.14% | Shared HTTP POST transport characteristics with identical port 80/443. |
| 3 | `Backdoor` | `Ransomware` | 8 | 3.48% | Boundary probability distribution near decision hyperplane. |
| 4 | `Fingerprinting` | `Port_Scanning` | 3 | 13.64% | Low-volume reconnaissance traffic closely mimicking normal query sweeps. |
| 5 | `Normal` | `Ransomware` | 2 | 0.37% | Boundary probability distribution near decision hyperplane. |

---

## 3. False Positive Analysis (`Normal` -> Attack)
- Total False Positives: 2
  - Sample 13402: Predicted as `Ransomware` with confidence 82.8%. Attributed to `len_payload`.
  - Sample 21674: Predicted as `Ransomware` with confidence 81.2%. Attributed to `len_payload`.

---

## 4. False Negative Analysis (Attack -> `Normal`)
- **Zero Dangerous Missed Threats**: Zero attacks were classified as Normal benign traffic.

---

## 5. Mitigation Strategies Implemented
1. **Class-Balanced Focal Loss**: Penalizes easy negatives while amplifying gradient contributions from minority confusions.
2. **Center Loss Metric Learning**: Forces threat vectors into tight spherical clusters in latent space, pulling apart ambiguous HTTP/Password flows.
3. **Calibrated Thresholds**: Adjusts decision boundary from rigid argmax to cost-sensitive rejection.