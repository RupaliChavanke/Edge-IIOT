"""
Page 16: Edge Hardware & Streaming Telemetry.
Live monitoring of edge system utilization (CPU/RAM/MPS), consumer lag, throughput, and latency waterfall.
"""

import psutil
import streamlit as st
import plotly.express as px

from evaluation.latency import profile_pipeline_latency
from evaluation.throughput import benchmark_streaming_rates
from visualization.plots import plot_latency_waterfall
from training.checkpoint import CheckpointManager
from preprocessing.loader import EdgeIIoTDataLoader
from models.proposed_model import ProposedHybridEdgeIIoTModel
import torch


def render_telemetry_page():
    st.title("📡 Edge Hardware & Streaming Telemetry")
    st.caption("Live profiling of edge resource consumption, distributed broker queue lag, and stage-by-stage microsecond latency.")

    # Hardware resource metrics
    cpu_pct = psutil.cpu_percent()
    ram_pct = psutil.virtual_memory().percent
    ram_used_gb = psutil.virtual_memory().used / (1024**3)
    ram_total_gb = psutil.virtual_memory().total / (1024**3)

    from dashboard.state import get_producer, get_consumer
    cons = get_consumer()
    prod = get_producer()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CPU Utilization", f"{cpu_pct:.1f}%")
    c2.metric("RAM Consumption", f"{ram_used_gb:.2f} / {ram_total_gb:.1f} GB ({ram_pct:.1f}%)")
    c3.metric("Producer Status", prod.get_stats()["status"] if prod else "STOPPED")
    cons_tput = cons.get_stats().get("messages_sec", 0.0) if cons and cons.is_running else 0.0
    c4.metric("Consumer Throughput", f"{cons_tput:.1f} msg/s")

    st.markdown("---")
    st.subheader("⏱️ Fine-Grained Latency Waterfall Decomposition")
    st.markdown(r"""
    $$T_{end\_to\_end} = T_{ingest} + T_{clean/scale} + T_{mRMR} + T_{CNN/Ghost} + T_{BiGRU} + T_{Attention/LowRank} + T_{publish}$$
    """)

    loader = EdgeIIoTDataLoader()
    if loader.load_pipeline("checkpoints").is_fitted:
        ckpt_mgr = CheckpointManager()
        model = ProposedHybridEdgeIIoTModel(input_dim=22, num_classes=len(loader.encoder.classes_))
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model, _ = ckpt_mgr.load_checkpoint(model, device=str(device))

        sample_event = {
            "tcp.dstport": 80.0, "tcp.srcport": 54321.0, "tcp.ack": 1024.0,
            "tcp.seq": 2048.0, "tcp.flags": 2.0, "tcp.len": 64.0
        }
        lat_profile = profile_pipeline_latency(model, loader, sample_event, num_iterations=40)
        fig_waterfall = plot_latency_waterfall(lat_profile)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        st.dataframe(pd.DataFrame(lat_profile).T, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Multi-Rate Streaming Throughput & Lag Benchmark")
    if st.button("🚀 Run Live Streaming Benchmark Sweep (10 - 500 msg/s)"):
        with st.spinner("Pacing messages across variable rate tiers..."):
            df_stream = benchmark_streaming_rates()
            st.dataframe(df_stream, use_container_width=True)

            fig_th = px.line(
                df_stream, x="Target_Rate_msg_s", y="Achieved_Throughput_msg_s",
                markers=True, title="Target vs Achieved Ingestion Throughput (events/sec)"
            )
            fig_th.update_layout(paper_bgcolor="#0F172A", plot_bgcolor="#1E293B", font=dict(color="#F8FAFC"))
            st.plotly_chart(fig_th, use_container_width=True)
