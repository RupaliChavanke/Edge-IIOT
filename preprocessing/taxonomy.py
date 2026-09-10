"""
Hierarchical Cybersecurity Attack Taxonomy Engine for Edge-IIoTset.
Implements 3-Level Security Hierarchy:
- Level 1: Binary Decision (Normal vs Attack)
- Level 2: Threat Category (DoS/DDoS, Information Gathering, MITM, Injection, Malware, Normal)
- Level 3: Exact Attack Type (15 classes)
Calculates mathematically consistent hierarchical probabilities, confidence tiers, and uncertainty detection.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np


# Level 3 to Level 2 (Threat Category) and Level 1 (Binary) Mapping
TAXONOMY_MAP = {
    # Benign
    "Normal": {
        "threat_category": "Normal",
        "binary_label": "Normal",
        "display_name": "Normal IIoT Traffic",
        "severity": "NORMAL"
    },
    # DoS / DDoS Category
    "DDoS_HTTP": {
        "threat_category": "DoS/DDoS",
        "binary_label": "Attack",
        "display_name": "DDoS HTTP Flood",
        "severity": "CRITICAL"
    },
    "DDoS_ICMP": {
        "threat_category": "DoS/DDoS",
        "binary_label": "Attack",
        "display_name": "DDoS ICMP Flood",
        "severity": "HIGH"
    },
    "DDoS_TCP": {
        "threat_category": "DoS/DDoS",
        "binary_label": "Attack",
        "display_name": "DDoS TCP SYN Flood",
        "severity": "CRITICAL"
    },
    "DDoS_UDP": {
        "threat_category": "DoS/DDoS",
        "binary_label": "Attack",
        "display_name": "DDoS UDP Flood",
        "severity": "HIGH"
    },
    # Information Gathering Category
    "Port_Scanning": {
        "threat_category": "Information Gathering",
        "binary_label": "Attack",
        "display_name": "Port Scanning",
        "severity": "MEDIUM"
    },
    "Fingerprinting": {
        "threat_category": "Information Gathering",
        "binary_label": "Attack",
        "display_name": "OS Fingerprinting",
        "severity": "LOW"
    },
    "Vulnerability_scanner": {
        "threat_category": "Information Gathering",
        "binary_label": "Attack",
        "display_name": "Vulnerability Scanner",
        "severity": "MEDIUM"
    },
    # MITM Category
    "MITM": {
        "threat_category": "MITM",
        "binary_label": "Attack",
        "display_name": "Man-In-The-Middle (ARP/DNS)",
        "severity": "CRITICAL"
    },
    # Injection Category
    "SQL_injection": {
        "threat_category": "Injection",
        "binary_label": "Attack",
        "display_name": "SQL Injection",
        "severity": "CRITICAL"
    },
    "XSS": {
        "threat_category": "Injection",
        "binary_label": "Attack",
        "display_name": "Cross-Site Scripting (XSS)",
        "severity": "HIGH"
    },
    "Uploading": {
        "threat_category": "Injection",
        "binary_label": "Attack",
        "display_name": "Malicious File Uploading",
        "severity": "CRITICAL"
    },
    # Malware Category
    "Backdoor": {
        "threat_category": "Malware",
        "binary_label": "Attack",
        "display_name": "Backdoor Trojan",
        "severity": "CRITICAL"
    },
    "Ransomware": {
        "threat_category": "Malware",
        "binary_label": "Attack",
        "display_name": "IIoT Ransomware",
        "severity": "CRITICAL"
    },
    "Password": {
        "threat_category": "Malware",
        "binary_label": "Attack",
        "display_name": "Password Brute Force",
        "severity": "HIGH"
    }
}


def get_taxonomy_categories() -> List[str]:
    """Returns list of unique Level 2 Threat Categories."""
    cats = sorted(list(set(info["threat_category"] for info in TAXONOMY_MAP.values())))
    return cats


def decode_hierarchical_prediction(
    class_probs: np.ndarray,
    class_names: List[str],
    entropy_val: float,
    route: str = "FAST",
    calibrated: bool = True,
    model_version: str = "v1.0",
    uncertainty_prob_threshold: float = 0.60,
    uncertainty_entropy_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Decodes full class probability vector into 3-tier security hierarchy:
    - Level 1: Binary (Normal vs Attack)
    - Level 2: Threat Category
    - Level 3: Exact Attack Type
    """
    class_probs = np.asarray(class_probs).flatten()
    prob_dict = {class_names[i]: float(class_probs[i]) for i in range(len(class_names))}

    # Level 3: Exact Attack Type
    winning_idx = int(np.argmax(class_probs))
    raw_attack_type = class_names[winning_idx]
    attack_confidence = float(class_probs[winning_idx])

    meta = TAXONOMY_MAP.get(raw_attack_type, {
        "threat_category": "Unknown",
        "binary_label": "Attack",
        "display_name": raw_attack_type,
        "severity": "HIGH"
    })
    exact_attack_display = meta["display_name"]
    threat_category = meta["threat_category"]
    binary_label = meta["binary_label"]

    # Level 2: Threat Category Probability
    category_probs: Dict[str, float] = {}
    for cname, p in prob_dict.items():
        cat = TAXONOMY_MAP.get(cname, {}).get("threat_category", "Unknown")
        category_probs[cat] = category_probs.get(cat, 0.0) + p

    threat_confidence = float(category_probs.get(threat_category, attack_confidence))

    # Level 1: Binary Decision Probability
    normal_prob = float(prob_dict.get("Normal", 0.0))
    attack_prob = float(1.0 - normal_prob)

    if binary_label == "Normal":
        binary_pred = "Normal"
        binary_conf = normal_prob
    else:
        binary_pred = "Attack"
        binary_conf = attack_prob

    # Confidence Tier Categorization
    if attack_confidence >= 0.95:
        confidence_tier = "VERY HIGH CONFIDENCE (≥0.95)"
    elif attack_confidence >= 0.80:
        confidence_tier = "HIGH CONFIDENCE (0.80-0.95)"
    elif attack_confidence >= 0.60:
        confidence_tier = "MEDIUM CONFIDENCE (0.60-0.80)"
    else:
        confidence_tier = "LOW CONFIDENCE (<0.60)"

    # Uncertainty Guardrail
    is_uncertain = (attack_confidence < uncertainty_prob_threshold) or (entropy_val > uncertainty_entropy_threshold)
    uncertainty_status = "UNCERTAIN / REQUIRES DEEP INSPECTION" if is_uncertain else "CONFIDENT DETECTION"

    return {
        "binary_prediction": binary_pred,
        "binary_confidence": round(binary_conf, 4),
        "threat_category": threat_category,
        "threat_confidence": round(threat_confidence, 4),
        "attack_type": exact_attack_display,
        "raw_attack_type": raw_attack_type,
        "attack_confidence": round(attack_confidence, 4),
        "confidence_tier": confidence_tier,
        "entropy": round(entropy_val, 4),
        "route": route,
        "calibrated": calibrated,
        "uncertainty": is_uncertain,
        "uncertain": is_uncertain,
        "uncertainty_status": uncertainty_status,
        "uncertainty_flag": uncertainty_status,
        "severity": meta["severity"],
        "model_version": model_version,
        "category_probabilities": {k: round(v, 4) for k, v in category_probs.items()}
    }
