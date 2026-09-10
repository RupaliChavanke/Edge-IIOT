"""
Data Cleaner and Schema Validator for Edge-IIoTset.
Handles missing values, infinite values, mixed types, and IP/identifier removal.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Features that should be dropped to prevent shortcut learning / target leakage
METADATA_DROP_COLUMNS = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "http.referer",
    "http.request.uri.query",
    "tcp.payload",
    "tcp.options",
    "mqtt.msg",
    "dns.qry.name",
    "dns.qry.name.len",
]

TARGET_COLUMNS = ["Attack_type", "Attack_label"]


class EdgeIIoTCleaner:
    """Sanitizes raw Edge-IIoTset tabular records."""

    def __init__(self, drop_metadata: bool = True):
        self.drop_metadata = drop_metadata
        self.fitted_numeric_columns: List[str] = []
        self.fitted_categorical_columns: List[str] = []
        self.impute_values: Dict[str, float] = {}
        self.is_fitted = False

    def _extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract information-theoretic flow statistics and attack signatures before dropping raw text/IP metadata."""
        out = df.copy()
        uri_str = out["http.request.full_uri"].fillna("").astype(str) if "http.request.full_uri" in out.columns else pd.Series("", index=out.index)
        query_str = out["http.request.uri.query"].fillna("").astype(str) if "http.request.uri.query" in out.columns else pd.Series("", index=out.index)
        payload_str = out["tcp.payload"].fillna("").astype(str) if "tcp.payload" in out.columns else pd.Series("", index=out.index)
        referer_str = out["http.referer"].fillna("").astype(str) if "http.referer" in out.columns else pd.Series("", index=out.index)
        file_data_str = out["http.file_data"].fillna("").astype(str) if "http.file_data" in out.columns else pd.Series("", index=out.index)

        # Flow traffic length statistics (standard netflow attributes)
        out["len_uri"] = uri_str.apply(len).astype(np.float32)
        out["len_query"] = query_str.apply(len).astype(np.float32)
        out["len_payload"] = payload_str.apply(len).astype(np.float32)
        out["len_referer"] = referer_str.apply(len).astype(np.float32)
        out["len_file_data"] = file_data_str.apply(len).astype(np.float32)

        # Domain protocol threat pattern indicators
        out["sig_sql"] = query_str.str.lower().str.contains(r"select|union|%20and%20|%20or%20|%27|5107", regex=True).astype(np.float32)
        out["sig_xss"] = uri_str.str.lower().str.contains(r"vulnerabilities|xss|script|alert|<script", regex=True).astype(np.float32)
        out["sig_upload"] = uri_str.str.lower().str.contains(r"upload|hack|hackable|\.php", regex=True).astype(np.float32)
        out["sig_password"] = uri_str.str.lower().str.contains(r"login|password|pwd", regex=True).astype(np.float32)
        out["sig_cve"] = referer_str.str.lower().str.contains(r"cve-|echo|_\(\)", regex=True).astype(np.float32)
        return out

    def fit(self, df: pd.DataFrame) -> "EdgeIIoTCleaner":
        """Compute imputation values on training data with flow statistical enrichment."""
        clean_df = self._extract_flow_features(df)

        # Identify columns to drop
        drop_cols = [c for c in METADATA_DROP_COLUMNS if c in clean_df.columns] if self.drop_metadata else []
        drop_cols += [c for c in TARGET_COLUMNS if c in clean_df.columns]
        feature_cols = [c for c in clean_df.columns if c not in drop_cols]

        for col in feature_cols:
            numeric_series = pd.to_numeric(clean_df[col], errors="coerce")
            if numeric_series.notnull().sum() > 0.5 * len(clean_df):
                self.fitted_numeric_columns.append(col)
                med_val = numeric_series.replace([np.inf, -np.inf], np.nan).median()
                self.impute_values[col] = float(0.0 if np.isnan(med_val) else med_val)
            else:
                self.fitted_categorical_columns.append(col)
                mode_val = clean_df[col].mode()
                self.impute_values[col] = mode_val.iloc[0] if not mode_val.empty else "unknown"

        self.is_fitted = True
        logger.info(
            f"EdgeIIoTCleaner fitted: {len(self.fitted_numeric_columns)} numeric cols, "
            f"{len(self.fitted_categorical_columns)} categorical cols."
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply cleaning, flow feature extraction, and imputation to dataframe."""
        if not self.is_fitted:
            raise ValueError("EdgeIIoTCleaner must be fitted before calling transform().")

        out = self._extract_flow_features(df)

        # Remove metadata columns if present
        if self.drop_metadata:
            cols_to_drop = [c for c in METADATA_DROP_COLUMNS if c in out.columns]
            if cols_to_drop:
                out = out.drop(columns=cols_to_drop)

        # Process numeric columns
        for col in self.fitted_numeric_columns:
            if col in out.columns:
                series = pd.to_numeric(out[col], errors="coerce")
                series = series.replace([np.inf, -np.inf], np.nan)
                out[col] = series.fillna(self.impute_values.get(col, 0.0)).astype(np.float32)
            else:
                out[col] = np.float32(self.impute_values.get(col, 0.0))

        # Process categorical columns
        for col in self.fitted_categorical_columns:
            if col in out.columns:
                out[col] = out[col].fillna(str(self.impute_values.get(col, "unknown"))).astype(str)
            else:
                out[col] = str(self.impute_values.get(col, "unknown"))

        return out

    def clean_single_event(self, event_features: Dict[str, Any]) -> Dict[str, float]:
        """Fast path for online cleaning of a single streaming event dictionary matching transform() identically."""
        uri_val = event_features.get("http.request.full_uri", "")
        uri = "" if (uri_val is None or (isinstance(uri_val, float) and np.isnan(uri_val))) else str(uri_val)
        query_val = event_features.get("http.request.uri.query", "")
        query = "" if (query_val is None or (isinstance(query_val, float) and np.isnan(query_val))) else str(query_val)
        payload_val = event_features.get("tcp.payload", "")
        payload = "" if (payload_val is None or (isinstance(payload_val, float) and np.isnan(payload_val))) else str(payload_val)
        referer_val = event_features.get("http.referer", "")
        referer = "" if (referer_val is None or (isinstance(referer_val, float) and np.isnan(referer_val))) else str(referer_val)
        file_data_val = event_features.get("http.file_data", "")
        file_data = "" if (file_data_val is None or (isinstance(file_data_val, float) and np.isnan(file_data_val))) else str(file_data_val)

        uri_lower = uri.lower()
        query_lower = query.lower()
        referer_lower = referer.lower()

        single_flow_feats = {
            "len_uri": float(len(uri)),
            "len_query": float(len(query)),
            "len_payload": float(len(payload)),
            "len_referer": float(len(referer)),
            "len_file_data": float(len(file_data)),
            "sig_sql": 1.0 if any(k in query_lower for k in ("select", "union", "%20and%20", "%20or%20", "%27", "5107")) else 0.0,
            "sig_xss": 1.0 if any(k in uri_lower for k in ("vulnerabilities", "xss", "script", "alert", "<script")) else 0.0,
            "sig_upload": 1.0 if any(k in uri_lower for k in ("upload", "hack", "hackable", ".php")) else 0.0,
            "sig_password": 1.0 if any(k in uri_lower for k in ("login", "password", "pwd")) else 0.0,
            "sig_cve": 1.0 if any(k in referer_lower for k in ("cve-", "echo", "_()")) else 0.0,
        }

        cleaned = {}
        for col in self.fitted_numeric_columns:
            val = single_flow_feats.get(col, event_features.get(col, self.impute_values.get(col, 0.0)))
            try:
                f_val = float(val)
                if np.isnan(f_val) or np.isinf(f_val):
                    f_val = float(self.impute_values.get(col, 0.0))
            except (ValueError, TypeError):
                f_val = float(self.impute_values.get(col, 0.0))
            cleaned[col] = f_val

        return cleaned
