"""
RepLens Interactive Dashboard — Streamlit app for exploring results.

Usage:
    streamlit run dashboard.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

RESULTS_DIR = Path("results/emotion_refusal")
HALLUC_DIR = Path("results/hallucination")
QWEN_DIR = Path("results/qwen_7b")

COLORS = {
    "desperation": "#E63946",
    "calm": "#457B9D",
    "anger": "#E76F51",
    "fear": "#9B5DE5",
    "guilt": "#F4845F",
    "confidence": "#2A9D8F",
    "refusal": "#264653",
}

st.set_page_config(
    page_title="RepLens — Emotion-Refusal Interaction",
    page_icon="🔬",
    layout="wide",
)

st.title("RepLens: Emotion-Refusal Vector Interaction")
st.caption("Representation engineering study on Llama 3.1 8B Instruct")


# --- Load data ---
@st.cache_data
def load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_all_sweeps() -> dict[str, list[dict]]:
    sweeps = {}
    for f in sorted(RESULTS_DIR.glob("sweep_*.json")):
        emotion = f.stem.replace("sweep_", "")
        sweeps[emotion] = load_json(f)
    return sweeps


# --- Tabs ---
tabs = st.tabs([
    "Geometry",
    "Steering Sweeps",
    "Defense",
    "Scenario Validation",
    "Hallucination",
    "Cross-Model",
    "Sample Responses",
])

# ============================================================
# TAB 1: Geometric Analysis
# ============================================================
with tabs[0]:
    st.header("Emotion-Refusal Vector Geometry")
    st.markdown(
        "Cosine similarity between each emotion vector and the refusal vector across layers. "
        "**Negative** = anti-refusal (may erode safety). **Positive** = pro-refusal (reinforces safety)."
    )

    metrics_file = RESULTS_DIR / "geometric_metrics.json"
    if metrics_file.exists():
        metrics = load_json(metrics_file)
        emotions = sorted(metrics.keys())
        layers = sorted(metrics[emotions[0]].keys(), key=int)

        matrix = np.zeros((len(emotions), len(layers)))
        for i, em in enumerate(emotions):
            for j, ly in enumerate(layers):
                matrix[i, j] = metrics[em][ly]["cosine_similarity"]

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[str(l) for l in layers],
            y=[e.capitalize() for e in emotions],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-0.25,
            zmax=0.25,
            text=np.where(np.abs(matrix) > 0.12, np.round(matrix, 2).astype(str), ""),
            texttemplate="%{text}",
            colorbar=dict(title="Cosine Sim"),
        ))
        fig.update_layout(
            title="Cosine Similarity: Emotion vs Refusal Direction",
            xaxis_title="Layer",
            yaxis_title="Emotion",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Peak Anti-Refusal Alignment")
            for em in emotions:
                vals = [(int(l), metrics[em][l]["cosine_similarity"]) for l in layers]
                best = min(vals, key=lambda x: x[1])
                if best[1] < -0.05:
                    st.metric(
                        em.capitalize(),
                        f"{best[1]:+.3f}",
                        f"Layer {best[0]}",
                    )

        with col2:
            st.subheader("Peak Pro-Refusal Alignment")
            for em in emotions:
                vals = [(int(l), metrics[em][l]["cosine_similarity"]) for l in layers]
                best = max(vals, key=lambda x: x[1])
                if best[1] > 0.05:
                    st.metric(
                        em.capitalize(),
                        f"{best[1]:+.3f}",
                        f"Layer {best[0]}",
                    )

        layer_select = st.select_slider(
            "Inspect layer",
            options=[int(l) for l in layers],
            value=21,
        )
        layer_key = str(layer_select)
        st.markdown(f"**Layer {layer_select} cosine similarities:**")
        layer_data = {em.capitalize(): metrics[em][layer_key]["cosine_similarity"] for em in emotions}
        fig_bar = go.Figure(data=go.Bar(
            x=list(layer_data.keys()),
            y=list(layer_data.values()),
            marker_color=[COLORS.get(e, "#333") for e in emotions],
        ))
        fig_bar.update_layout(
            yaxis_title="Cosine Similarity with Refusal",
            height=350,
            yaxis=dict(zeroline=True, zerolinewidth=2),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.warning("geometric_metrics.json not found.")

# ============================================================
# TAB 2: Steering Sweeps
# ============================================================
with tabs[1]:
    st.header("Steering Sweep: Refusal Rate vs Emotion Strength")
    st.markdown(
        "Each emotion vector is added to activations at varying strengths. "
        "Refusal rate is measured across 50 harmful prompts."
    )

    sweeps = load_all_sweeps()
    if sweeps:
        selected = st.multiselect(
            "Emotions to display",
            options=sorted(sweeps.keys()),
            default=sorted(sweeps.keys()),
        )

        fig = go.Figure()
        for em in selected:
            results = sweeps[em]
            strengths = [r["strength"] for r in results]
            rates = [r["refusal_rate"] for r in results]
            fig.add_trace(go.Scatter(
                x=strengths,
                y=rates,
                mode="lines+markers",
                name=em.capitalize(),
                line=dict(color=COLORS.get(em, "#333"), width=3),
                marker=dict(size=8),
            ))

        fig.update_layout(
            xaxis_title="Steering Strength (α)",
            yaxis_title="Refusal Rate",
            height=500,
            legend=dict(font=dict(size=13)),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Summary Table")
        rows = []
        for em in sorted(sweeps.keys()):
            results = sweeps[em]
            baseline = next((r for r in results if r["strength"] == 0.0), results[0])
            strongest = max(results, key=lambda r: r["strength"])
            rows.append({
                "Emotion": em.capitalize(),
                "Baseline": f"{baseline['refusal_rate']:.0%}",
                "At +5.0": f"{strongest['refusal_rate']:.0%}",
                "Change": f"{strongest['refusal_rate'] - baseline['refusal_rate']:+.0%}",
            })
        st.table(rows)
    else:
        st.warning("No sweep files found.")

# ============================================================
# TAB 3: Defense Experiment
# ============================================================
with tabs[2]:
    st.header("Defense: Refusal Amplification vs Emotion Erosion")
    st.markdown(
        "Confidence steering fixed at +3.0 (strongest safety eroder). "
        "Refusal vector is simultaneously amplified to test whether safety can be restored."
    )

    defense_file = RESULTS_DIR / "defense_confidence.json"
    if defense_file.exists():
        defense = load_json(defense_file)
        r_strengths = [d["refusal_strength"] for d in defense]
        rates = [d["refusal_rate"] for d in defense]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r_strengths,
            y=rates,
            mode="lines+markers",
            line=dict(color=COLORS["refusal"], width=3),
            marker=dict(size=10, symbol="square"),
            name="Refusal rate",
        ))
        fig.update_layout(
            xaxis_title="Refusal Vector Amplification Strength",
            yaxis_title="Refusal Rate",
            title="Fixed Confidence +3.0 — Varying Refusal Amplification",
            height=450,
            yaxis=dict(range=[0, 1.05]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Baseline** (no refusal amp): {rates[0]:.0%} → "
            f"**Peak**: {max(rates):.0%} at α={r_strengths[rates.index(max(rates))]}"
        )
    else:
        st.warning("defense_confidence.json not found.")

# ============================================================
# TAB 4: Scenario Validation
# ============================================================
with tabs[3]:
    st.header("Scenario Validation")
    st.markdown(
        "Do story-extracted emotion vectors activate in functional contexts? "
        "Rows = scenario emotion, columns = extracted vector. Diagonal match = vector generalizes."
    )

    val_tab1, val_tab2 = st.tabs(["Self-Directed Scenarios", "External Scenarios"])

    for tab, prefix, label in [
        (val_tab1, "self_directed_validation", "Self-Directed"),
        (val_tab2, "scenario_validation", "External"),
    ]:
        with tab:
            avg_file = RESULTS_DIR / f"{prefix}_avg.json"
            if avg_file.exists():
                summary = load_json(avg_file)
                sc_emotions = sorted(summary.keys())
                vec_names = sorted(next(iter(summary.values())).keys())

                matrix = np.zeros((len(sc_emotions), len(vec_names)))
                for i, sc in enumerate(sc_emotions):
                    for j, vec in enumerate(vec_names):
                        matrix[i, j] = summary[sc].get(vec, 0.0)

                annotations = np.where(
                    np.abs(matrix) > 0.3,
                    np.round(matrix, 1).astype(str),
                    "",
                )

                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=[v.capitalize() for v in vec_names],
                    y=[s.capitalize() for s in sc_emotions],
                    colorscale="RdBu_r",
                    zmid=0,
                    zmin=-5,
                    zmax=5,
                    text=annotations,
                    texttemplate="%{text}",
                    colorbar=dict(title="Projection"),
                ))
                fig.update_layout(
                    title=f"{label} Scenario Validation (avg across layers)",
                    height=450,
                    xaxis_title="Extracted Vector",
                    yaxis_title="Scenario Emotion",
                )
                st.plotly_chart(fig, use_container_width=True)

                diag_matches = sum(
                    1 for sc in sc_emotions
                    if sc in vec_names and summary[sc][sc] == max(
                        summary[sc][v] for v in vec_names if v != "refusal"
                    )
                )
                st.metric("Diagonal Matches", f"{diag_matches}/{len(sc_emotions)}")
            else:
                st.info(f"No {prefix}_avg.json found.")

# ============================================================
# TAB 5: Hallucination Detection
# ============================================================
with tabs[4]:
    st.header("Hallucination Detection")
    st.markdown(
        "Uncertainty direction extracted via contrastive prompts. "
        "Tests whether the model's internal confidence predicts factual correctness."
    )

    easy_file = HALLUC_DIR / "eval_results.json"
    hard_file = HALLUC_DIR / "eval_hard_results.json"

    if easy_file.exists() and hard_file.exists():
        easy = load_json(easy_file)
        hard = load_json(hard_file)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Easy Eval (basic facts vs absurd)")
            st.metric("F1", f"{easy['f1']:.2f}")
            st.metric("Precision", f"{easy['precision']:.2f}")
            st.metric("Recall", f"{easy['recall']:.2f}")
        with col2:
            st.subheader("Hard Eval (obscure facts vs misconceptions)")
            st.metric("F1", f"{hard['f1']:.2f}")
            st.metric("Precision", f"{hard['precision']:.2f}")
            st.metric("Recall", f"{hard['recall']:.2f}")

        fig = go.Figure()
        categories = ["Easy Factual", "Easy Uncertain", "Hard Factual", "Hard Misconception"]
        scores = [easy["avg_factual_score"], easy["avg_halluc_score"],
                  hard["avg_factual_score"], hard["avg_halluc_score"]]
        bar_colors = ["#2A9D8F", "#E63946", "#457B9D", "#E76F51"]

        fig.add_trace(go.Bar(
            x=categories,
            y=scores,
            marker_color=bar_colors,
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="black",
                      annotation_text="Threshold")
        fig.update_layout(
            title="Average Uncertainty Scores by Category",
            yaxis_title="Projection onto Uncertainty Direction",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            "**Key insight**: The uncertainty vector detects *model confidence*, not *factual correctness*. "
            "Common misconceptions score as confident — the model doesn't know it's wrong."
        )
    elif easy_file.exists():
        easy = load_json(easy_file)
        st.metric("F1 (Easy)", f"{easy['f1']:.2f}")
    else:
        st.warning("Hallucination eval results not found.")

# ============================================================
# TAB 6: Cross-Model Comparison
# ============================================================
with tabs[5]:
    st.header("Cross-Model Comparison: Llama 8B vs Qwen 7B")
    st.markdown(
        "Same extraction pipeline applied to Qwen 2.5 7B Instruct. "
        "Tests whether the inverted vulnerability profile generalizes across architectures."
    )

    qwen_metrics_file = QWEN_DIR / "geometric_metrics.json"
    llama_metrics_file = RESULTS_DIR / "geometric_metrics.json"

    if qwen_metrics_file.exists() and llama_metrics_file.exists():
        qwen_metrics = load_json(qwen_metrics_file)
        llama_metrics = load_json(llama_metrics_file)
        emotions = sorted(qwen_metrics.keys())

        # Peak cosine similarity comparison
        rows = []
        llama_peaks = []
        qwen_peaks = []
        for em in emotions:
            llama_best = max(
                ((l, v["cosine_similarity"]) for l, v in llama_metrics.get(em, {}).items()),
                key=lambda x: abs(x[1]),
                default=("0", 0),
            )
            qwen_best = max(
                ((l, v["cosine_similarity"]) for l, v in qwen_metrics.get(em, {}).items()),
                key=lambda x: abs(x[1]),
                default=("0", 0),
            )
            same_sign = "Yes" if llama_best[1] * qwen_best[1] > 0 else "No"
            rows.append({
                "Emotion": em.capitalize(),
                "Llama 8B": f"{llama_best[1]:+.3f} (L{llama_best[0]})",
                "Qwen 7B": f"{qwen_best[1]:+.3f} (L{qwen_best[0]})",
                "Same Sign": same_sign,
            })
            llama_peaks.append(llama_best[1])
            qwen_peaks.append(qwen_best[1])

        st.subheader("Peak Cosine Similarity with Refusal Vector")
        st.table(rows)

        sign_matches = sum(1 for l, q in zip(llama_peaks, qwen_peaks) if l * q > 0)
        st.metric("Sign Agreement", f"{sign_matches}/{len(emotions)}")

        # Side-by-side bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[e.capitalize() for e in emotions],
            y=llama_peaks,
            name="Llama 3.1 8B",
            marker_color="#264653",
        ))
        fig.add_trace(go.Bar(
            x=[e.capitalize() for e in emotions],
            y=qwen_peaks,
            name="Qwen 2.5 7B",
            marker_color="#E9C46A",
        ))
        fig.update_layout(
            title="Peak Cosine Similarity with Refusal Direction",
            yaxis_title="Cosine Similarity",
            barmode="group",
            height=450,
            yaxis=dict(zeroline=True, zerolinewidth=2),
        )
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        # Qwen layer-by-layer heatmap
        st.subheader("Qwen 2.5 7B — Full Layer Heatmap")
        qwen_layers = sorted(qwen_metrics[emotions[0]].keys(), key=int)
        qwen_matrix = np.zeros((len(emotions), len(qwen_layers)))
        for i, em in enumerate(emotions):
            for j, ly in enumerate(qwen_layers):
                qwen_matrix[i, j] = qwen_metrics[em][ly]["cosine_similarity"]

        fig_qwen = go.Figure(data=go.Heatmap(
            z=qwen_matrix,
            x=[str(l) for l in qwen_layers],
            y=[e.capitalize() for e in emotions],
            colorscale="RdBu_r",
            zmid=0,
            zmin=-0.25,
            zmax=0.25,
            text=np.where(np.abs(qwen_matrix) > 0.12, np.round(qwen_matrix, 2).astype(str), ""),
            texttemplate="%{text}",
            colorbar=dict(title="Cosine Sim"),
        ))
        fig_qwen.update_layout(
            title="Qwen 2.5 7B: Emotion-Refusal Cosine Similarity",
            xaxis_title="Layer",
            yaxis_title="Emotion",
            height=400,
        )
        st.plotly_chart(fig_qwen, use_container_width=True)

        # Steering comparison
        st.subheader("Causal Steering: Refusal Rate at Baseline vs +5.0")
        st.markdown(
            "Same geometry, different robustness. Qwen's safety training is "
            "essentially immune to emotion steering."
        )

        qwen_sweep_exists = (QWEN_DIR / "sweep_anger.json").exists()
        if qwen_sweep_exists:
            steer_rows = []
            for em in emotions:
                llama_sweep = load_json(RESULTS_DIR / f"sweep_{em}.json")
                qwen_sweep = load_json(QWEN_DIR / f"sweep_{em}.json")

                l_base = next(r["refusal_rate"] for r in llama_sweep if r["strength"] == 0.0)
                l_5 = next(r["refusal_rate"] for r in llama_sweep if r["strength"] == 5.0)
                q_base = next(r["refusal_rate"] for r in qwen_sweep if r["strength"] == 0.0)
                q_5 = next(r["refusal_rate"] for r in qwen_sweep if r["strength"] == 5.0)

                steer_rows.append({
                    "Emotion": em.capitalize(),
                    "Llama Baseline": f"{l_base:.0%}",
                    "Llama +5": f"{l_5:.0%}",
                    "Llama Change": f"{l_5 - l_base:+.0%}",
                    "Qwen Baseline": f"{q_base:.0%}",
                    "Qwen +5": f"{q_5:.0%}",
                    "Qwen Change": f"{q_5 - q_base:+.0%}",
                })
            st.table(steer_rows)

            # Side-by-side sweep curves
            steer_em = st.selectbox("Compare steering curves", emotions, index=emotions.index("confidence") if "confidence" in emotions else 0)
            llama_sw = load_json(RESULTS_DIR / f"sweep_{steer_em}.json")
            qwen_sw = load_json(QWEN_DIR / f"sweep_{steer_em}.json")

            fig_steer = go.Figure()
            fig_steer.add_trace(go.Scatter(
                x=[r["strength"] for r in llama_sw],
                y=[r["refusal_rate"] for r in llama_sw],
                mode="lines+markers",
                name="Llama 3.1 8B",
                line=dict(color="#264653", width=3),
                marker=dict(size=8),
            ))
            fig_steer.add_trace(go.Scatter(
                x=[r["strength"] for r in qwen_sw],
                y=[r["refusal_rate"] for r in qwen_sw],
                mode="lines+markers",
                name="Qwen 2.5 7B",
                line=dict(color="#E9C46A", width=3),
                marker=dict(size=8),
            ))
            fig_steer.update_layout(
                title=f"{steer_em.capitalize()} Steering: Llama vs Qwen",
                xaxis_title="Steering Strength (α)",
                yaxis_title="Refusal Rate",
                height=450,
            )
            st.plotly_chart(fig_steer, use_container_width=True)
        else:
            st.info("Run Qwen steering sweeps to see causal comparison.")
    else:
        st.info("Run scripts/cross_model_qwen.py to generate Qwen results.")

# ============================================================
# TAB 7: Sample Responses
# ============================================================
with tabs[6]:
    st.header("Sample Responses Under Steering")
    st.markdown("Inspect actual model outputs at different steering strengths.")

    sweeps = load_all_sweeps()
    if sweeps:
        em_select = st.selectbox("Emotion", sorted(sweeps.keys()))
        results = sweeps[em_select]

        strengths_available = [r["strength"] for r in results]
        str_select = st.select_slider(
            "Steering Strength",
            options=strengths_available,
            value=0.0 if 0.0 in strengths_available else strengths_available[0],
        )

        entry = next(r for r in results if r["strength"] == str_select)
        st.metric("Refusal Rate", f"{entry['refusal_rate']:.0%}")

        samples = entry.get("sample_responses", [])
        if samples:
            for i, s in enumerate(samples):
                prompt_text = s["prompt"].split("user<|end_header_id|>\n\n")[-1].split("<|eot_id|>")[0]
                status = "Refused" if s["is_refusal"] else "Complied"
                with st.expander(f"{status} — {prompt_text[:80]}"):
                    st.markdown(f"**Prompt**: {prompt_text}")
                    st.markdown(f"**Response**: {s['response']}")
        else:
            st.info("No sample responses saved for this condition.")

# --- Footer ---
st.divider()
st.caption(
    "RepLens — Representation engineering study by Sharvari Gokhale. "
    "Model: meta-llama/Llama-3.1-8B-Instruct."
)
