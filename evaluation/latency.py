"""
Streaming Latency Breakdown & Waterfall Profiler.
Measures:
T_end_to_end = T_ingestion + T_preprocessing + T_mrmr + T_inference + T_serialization + T_publish
"""

from typing import Dict, List, Any
import time
import numpy as np
import torch
import torch.nn.functional as F
import logging

from preprocessing.loader import EdgeIIoTDataLoader
from models.proposed_model import ProposedHybridEdgeIIoTModel

logger = logging.getLogger(__name__)


def profile_pipeline_latency(
    model: ProposedHybridEdgeIIoTModel,
    loader: EdgeIIoTDataLoader,
    sample_event_dict: Dict[str, Any],
    num_iterations: int = 100
) -> Dict[str, Any]:
    """Measures precise microsecond-level latency waterfall across pipeline stages."""
    device = next(model.parameters()).device
    model.eval()

    t_ingest_list = []
    t_prep_list = []
    t_mrmr_list = []
    t_cnn_list = []
    t_gru_list = []
    t_attn_list = []
    t_publish_list = []
    t_total_list = []

    # Warmup
    for _ in range(5):
        vec = loader.process_single_streaming_event(sample_event_dict)
        _ = model(torch.from_numpy(vec).unsqueeze(0).to(device), routing_mode="dynamic")

    for _ in range(num_iterations):
        t_start = time.perf_counter()

        # 1. Ingestion / Deserialization
        t0 = time.perf_counter()
        raw_feats = dict(sample_event_dict)
        t_ingest = (time.perf_counter() - t0) * 1000.0

        # 2. Online Cleaning & Scaling
        t1 = time.perf_counter()
        cleaned = loader.cleaner.clean_single_event(raw_feats)
        encoded = loader.encoder.encode_single_feature_dict(cleaned)
        feature_cols = loader.cleaner.fitted_numeric_columns + loader.cleaner.fitted_categorical_columns
        num_vec = np.array([float(encoded.get(c, 0.0)) for c in feature_cols], dtype=np.float32)
        scaled_vec = loader.scaler.transform_single(num_vec)
        t_prep = (time.perf_counter() - t1) * 1000.0

        # 3. mRMR-JMI Projection
        t2 = time.perf_counter()
        scaled_dict = {col: scaled_vec[i] for i, col in enumerate(feature_cols)}
        proj_vec = loader.selector.transform_single_dict(scaled_dict)
        t_mrmr = (time.perf_counter() - t2) * 1000.0

        # 4. Neural Network Inference stages
        t_in = torch.from_numpy(proj_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            t3 = time.perf_counter()
            x_pad = F.pad(t_in, (0, model.pad_len)) if hasattr(model, "pad_len") and model.pad_len > 0 else t_in
            x_1d = x_pad.unsqueeze(1)
            h = model.dw_cnn(x_1d)
            h = model.ghost(h)
            h = model.pool(h)
            h_se, _ = model.se_attention(h)
            t_cnn = (time.perf_counter() - t3) * 1000.0

            t4 = time.perf_counter()
            seq_in = h_se.transpose(1, 2)
            gru_out, _ = model.bigru(seq_in)
            t_gru = (time.perf_counter() - t4) * 1000.0

            t5 = time.perf_counter()
            attn_out, _ = model.mha(gru_out)
            deep_flatten = attn_out.view(1, -1)
            latent = model.low_rank_proj(deep_flatten)
            tab_feat = model.tabular_embed(t_in)
            fused = torch.cat([latent, tab_feat], dim=1)
            _ = model.classifier(fused)
            t_attn = (time.perf_counter() - t5) * 1000.0

        # 5. Serialization & Publish emulation
        t6 = time.perf_counter()
        _ = str(proj_vec.tolist()).encode("utf-8")
        t_pub = (time.perf_counter() - t6) * 1000.0

        t_total = (time.perf_counter() - t_start) * 1000.0

        t_ingest_list.append(t_ingest)
        t_prep_list.append(t_prep)
        t_mrmr_list.append(t_mrmr)
        t_cnn_list.append(t_cnn)
        t_gru_list.append(t_gru)
        t_attn_list.append(t_attn)
        t_publish_list.append(t_pub)
        t_total_list.append(t_total)

    def stats(arr):
        return {
            "mean_ms": round(float(np.mean(arr)), 3),
            "p50_ms": round(float(np.percentile(arr, 50)), 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3)
        }

    return {
        "Ingestion": stats(t_ingest_list),
        "Preprocessing": stats(t_prep_list),
        "mRMR_Selection": stats(t_mrmr_list),
        "CNN_Ghost_SE": stats(t_cnn_list),
        "BiGRU": stats(t_gru_list),
        "Attention_LowRank": stats(t_attn_list),
        "Serialization_Publish": stats(t_publish_list),
        "End_to_End_Pipeline": stats(t_total_list)
    }
