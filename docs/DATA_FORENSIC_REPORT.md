# Dataset Forensic Audit Report: Edge-IIoTset

**Dataset File**: `data/samples/edge_iiot_sample.csv`  
**Total Samples**: 23,548  
**Total Columns**: 63 (61 candidate features + `Attack_label` + `Attack_type`)  

---

## 1. Summary Statistics & Data Hygiene

| Metric | Count / Value | Details |
| :--- | :--- | :--- |
| **Total Rows** | 23,548 | Full experimental dataset |
| **Exact Duplicate Rows** | 0 | Identical across all 63 columns |
| **Feature-Only Duplicates** | 0 | Identical feature vectors |
| **Numerical Features** | 57 | Convertible to continuous float |
| **Categorical / String Features** | 4 | Text, IP addresses, payloads, protocol names |
| **Constant Columns (0 variance)** | 9 | icmp.unused, http.tls_port, dns.qry.type, dns.retransmit_request, dns.retransmit_request_in, mqtt.msg_decoded_as, mbtcp.len, mbtcp.trans_id, mbtcp.unit_id |
| **Near-Constant (2 unique vals)** | 12 | 12 columns (flags, binary indicators) |
| **High-Cardinality Columns (>500 unique)** | 15 | Timestamps, IPs, raw ports, sequence numbers |
| **Infinite Values Found** | 0 | Present in 0 columns prior to cleaning |

---

## 2. Attack and Class Distribution

### Multiclass Distribution (`Attack_type`)

| Attack Type | Sample Count | Percentage | Telemetry Category |
| :--- | :--- | :--- | :--- |
| `Normal` | 3,645 | 15.48% | Benign Normal |
| `DDoS_UDP` | 2,175 | 9.24% | Bulk Attack |
| `DDoS_ICMP` | 2,113 | 8.97% | Bulk Attack |
| `Ransomware` | 1,639 | 6.96% | Bulk Attack |
| `DDoS_HTTP` | 1,584 | 6.73% | Bulk Attack |
| `SQL_injection` | 1,546 | 6.57% | Bulk Attack |
| `Uploading` | 1,541 | 6.54% | Bulk Attack |
| `DDoS_TCP` | 1,537 | 6.53% | Bulk Attack |
| `Backdoor` | 1,529 | 6.49% | Bulk Attack |
| `Port_Scanning` | 1,511 | 6.42% | Bulk Attack |
| `Vulnerability_scanner` | 1,511 | 6.42% | Bulk Attack |
| `XSS` | 1,508 | 6.40% | Bulk Attack |
| `Password` | 1,499 | 6.37% | Bulk Attack |
| `Fingerprinting` | 150 | 0.64% | Minority Attack |
| `MITM` | 60 | 0.25% | Minority Attack |

### Binary Distribution (`Attack_label`)

- **Attack (1)**: 19,903 (84.52%)
- **Normal (0)**: 3,645 (15.48%)

---

## 3. Data Leakage & Shortcut Learning Analysis

### A. Provenance & IP Shortcut Leakage
- **Total Source IPs (`ip.src_host`)**: 2,856
- **IPs Appearing in Exactly ONE Attack Class**: 2,852 (99.86%)
- **Forensic Proof of Shortcut Learning**: In Edge-IIoTset, specific testbed host IP addresses (e.g. `192.168.0.101` which is 100% Normal) correlate directly with whether an attack was launched from that virtual machine. If `ip.src_host` or `ip.dst_host` is encoded as a feature, machine learning models achieve >99.8% accuracy simply by memorizing IP addresses, without learning any packet dynamics.

### B. Temporal Serialization Artifacts (`frame.time`)
- `frame.time` consists of sequential timestamps recorded during physical testbed execution.
- Randomly splitting rows allows packets from the exact same sub-second attack burst to be partitioned across both training and test sets, causing severe temporal cross-split contamination.

### C. Hand-Engineered String Signatures vs. NetFlow Features
- Previous scripts extracted string regex indicators (`sig_sql`, `sig_xss`, `sig_upload`, `sig_password`, `sig_cve`) from raw L7 URL queries.
- While valid for Layer 7 Deep Packet Inspection (DPI), in a flow-based IDS these act as deterministic signature rules rather than generalizable machine learning representations.

---

## 4. Single-Feature Predictability & Mutual Information Ranking

| Feature Name | Mutual Information | Single-Feature DT Acc | Single-Feature Macro-F1 | Status / Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| `tcp.dstport` | 1.3967 | 29.37% | 9.21% | ✅ RETAIN |
| `tcp.srcport` | 1.3428 | 27.48% | 9.53% | ✅ RETAIN |
| `tcp.ack` | 1.3153 | 28.44% | 12.40% | ✅ RETAIN |
| `tcp.seq` | 1.0150 | 31.68% | 16.09% | ✅ RETAIN |
| `tcp.flags` | 0.8061 | 29.74% | 12.04% | ✅ RETAIN |
| `tcp.ack_raw` | 0.7846 | 25.73% | 12.00% | ✅ RETAIN |
| `tcp.len` | 0.6726 | 25.24% | 12.31% | ✅ RETAIN |
| `tcp.checksum` | 0.4429 | 24.60% | 6.27% | ✅ RETAIN |
| `udp.stream` | 0.3200 | 24.71% | 8.61% | ✅ RETAIN |
| `tcp.flags.ack` | 0.3100 | 22.28% | 4.91% | ✅ RETAIN |
| `icmp.checksum` | 0.3047 | 24.45% | 8.47% | ✅ RETAIN |
| `icmp.seq_le` | 0.2957 | 24.55% | 10.37% | ✅ RETAIN |
| `http.content_length` | 0.1799 | 20.94% | 8.49% | ✅ RETAIN |
| `tcp.payload` | 0.1258 | 19.47% | 13.21% | ⚠️ DROP (Metadata Leak) |
| `tcp.connection.rst` | 0.1191 | 16.88% | 4.22% | ✅ RETAIN |
| `tcp.connection.syn` | 0.0885 | 18.47% | 4.51% | ✅ RETAIN |
| `http.response` | 0.0718 | 18.15% | 5.06% | ✅ RETAIN |
| `mqtt.msgtype` | 0.0688 | 15.48% | 1.79% | ✅ RETAIN |
| `mqtt.hdrflags` | 0.0651 | 15.48% | 1.79% | ✅ RETAIN |
| `tcp.connection.fin` | 0.0549 | 15.48% | 1.79% | ✅ RETAIN |
| `mqtt.len` | 0.0457 | 15.48% | 1.79% | 🔍 CANDIDATE |
| `dns.qry.name` | 0.0352 | 16.49% | 7.66% | ⚠️ DROP (Metadata Leak) |
| `tcp.options` | 0.0319 | 15.73% | 8.34% | ⚠️ DROP (Metadata Leak) |
| `udp.time_delta` | 0.0286 | 15.71% | 8.12% | 🔍 CANDIDATE |
| `tcp.connection.synack` | 0.0231 | 15.48% | 1.79% | 🔍 CANDIDATE |

---

## 5. Scientifically Defensible Feature Recommendation

### Features Excluded to Prevent Target Leakage:
1. `frame.time`: Temporal burst correlation.
2. `ip.src_host`, `ip.dst_host`: Static testbed host IP memorization.
3. `arp.src.proto_ipv4`, `arp.dst.proto_ipv4`: ARP IP addresses.
4. `http.file_data`, `http.request.full_uri`, `http.referer`, `http.request.uri.query`: Text payloads prone to exact string memorization.
5. `tcp.payload`, `tcp.options`: Raw payload byte dumps.
6. `mqtt.msg`: Raw MQTT sensor string values.
7. `dns.qry.name`, `dns.qry.name.len`: Domain query string names.
8. All 9 Zero-Variance Constant Columns (`icmp.unused`, `http.tls_port`, `dns.qry.type`, etc.).

### Approved Leakage-Free Feature Set:
- Pure transport & network flow attributes: TCP/UDP ports, TCP flag bits (`syn`, `ack`, `fin`, `rst`), packet length stats (`tcp.len`), sequence/acknowledgment numbers, ICMP checksum & sequence, DNS query flags, MQTT header attributes.
- Flow-level length attributes: `len_payload`, `len_uri`, `len_query` (numerical byte lengths, without string content).

---

## 6. Forensic Conclusion
The forensic audit conclusively proves that naive evaluations on Edge-IIoTset reporting >99% accuracy suffer from IP address and timestamp leakage. A true, leakage-free benchmark must completely purge metadata identifiers and enforce strictly isolated train/val/test splits.