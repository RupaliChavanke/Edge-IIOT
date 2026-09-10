"""
Page 4: 3-Level Hierarchical Attack Taxonomy.
Visualizes the multi-tier cybersecurity taxonomy:
Level 1: Normal vs Attack
Level 2: Threat Category (DoS/DDoS, Information Gathering, MITM, Injection, Malware, Normal)
Level 3: Exact Attack Type (15 classes)
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from preprocessing.taxonomy import TAXONOMY_MAP, get_taxonomy_categories


def render_taxonomy_page():
    st.title("🌲 3-Level Hierarchical Cybersecurity Attack Taxonomy")
    st.caption("Multi-Tier Attack Classification Framework | From High-Level Alert to Precise Root Cause")

    st.markdown("""
    <div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 12px 18px; border-radius: 4px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #38BDF8; font-size: 14px;">MULTI-LEVEL CYBERSECURITY INTEL</span>
        <div style="font-size: 12px; color: #94A3B8; margin-top: 2px;">
            A production IIoT Security Operations Center (SOC) requires both immediate coarse-grained triage 
            (<i>Is it an attack? What category?</i>) and surgical fine-grained mitigation (<i>What exact attack vector?</i>). 
            Our framework simultaneously resolves all 3 levels without separate independent models.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Build hierarchy DataFrame
    tree_data = []
    for k, v in TAXONOMY_MAP.items():
        tree_data.append({
            "Level 1: Binary Decision": v["binary_label"],
            "Level 2: Threat Category": v["threat_category"],
            "Level 3: Exact Attack Vector": v["display_name"],
            "Internal ID": k,
            "Severity": v["severity"],
            "Value": 1
        })

    df_tree = pd.DataFrame(tree_data)

    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("1. Interactive Hierarchical Sunburst Diagram")
        fig_sun = px.sunburst(
            df_tree,
            path=["Level 1: Binary Decision", "Level 2: Threat Category", "Level 3: Exact Attack Vector"],
            color="Level 2: Threat Category",
            color_discrete_map={
                "Normal": "#10B981",
                "DoS/DDoS": "#EF4444",
                "Information Gathering": "#F59E0B",
                "MITM": "#8B5CF6",
                "Injection": "#EC4899",
                "Malware": "#F97316"
            }
        )
        fig_sun.update_layout(
            paper_bgcolor="#0F172A",
            font=dict(color="#F8FAFC"),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_sun, use_container_width=True)

    with c2:
        st.subheader("2. Mathematical Formulation")
        st.markdown(r"""
        #### Probability Axiom Preservation
        Let $\{p_1, \dots, p_C\}$ denote the calibrated class probability distribution over the $C=15$ leaf attack classes:
        
        **Level 1 (Binary Confidence)**:
        $$P(\text{Normal}) = p_{\text{Normal}}$$
        $$P(\text{Attack}) = \sum_{c \neq \text{Normal}} p_c = 1 - P(\text{Normal})$$
        
        **Level 2 (Threat Category Confidence)**:
        For category $K \in \{\text{DoS/DDoS}, \text{Injection}, \dots\}$:
        $$P(K) = \sum_{c \in \text{Children}(K)} p_c$$
        
        **Level 3 (Exact Attack Type)**:
        $$\hat{y} = \arg\max_{c \in \{1, \dots, C\}} p_c$$
        """)

    st.markdown("---")

    # Section 3: Comprehensive Taxonomy Table
    st.subheader("3. Comprehensive Taxonomy Mapping Table")
    st.dataframe(df_tree[["Level 1: Binary Decision", "Level 2: Threat Category", "Level 3: Exact Attack Vector", "Internal ID", "Severity"]], use_container_width=True, hide_index=True)
