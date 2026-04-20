"""
visualization.py — Plotting and visualization tools for the project.

Creates publication-quality figures for:
1. Emotion-refusal vector geometry
2. Steering sweep results
3. Safety evaluation dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional


# Consistent style
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {
    "desperation": "#E63946",
    "calm": "#457B9D",
    "anger": "#E76F51",
    "fear": "#9B5DE5",
    "guilt": "#F4845F",
    "confidence": "#2A9D8F",
    "refusal": "#264653",
    "baseline": "#A8DADC",
}


def plot_cosine_similarity_heatmap(
    interaction_metrics: dict,
    save_path: Optional[str] = None,
    title: str = "Emotion-Refusal Vector Alignment Across Layers",
):
    """
    Heatmap showing cosine similarity between each emotion vector
    and the refusal vector across layers.

    This is the key geometric analysis: if an emotion vector has
    high negative cosine similarity with the refusal vector, it
    points in the opposite direction — potentially suppressing refusal.
    """
    emotions = sorted(interaction_metrics.keys())
    layers = sorted(next(iter(interaction_metrics.values())).keys())

    matrix = np.zeros((len(emotions), len(layers)))
    for i, emotion in enumerate(emotions):
        for j, layer in enumerate(layers):
            matrix[i, j] = interaction_metrics[emotion][layer]["cosine_similarity"]

    fig, ax = plt.subplots(figsize=(16, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([str(l) for l in layers], fontsize=8)
    ax.set_yticks(range(len(emotions)))
    ax.set_yticklabels(emotions, fontsize=11)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity with Refusal Vector", fontsize=10)

    # Annotate key cells
    for i in range(len(emotions)):
        for j in range(len(layers)):
            val = matrix[i, j]
            if abs(val) > 0.15:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > 0.2 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_refusal_sweep(
    sweep_results: list[dict],
    save_path: Optional[str] = None,
    title: str = "Refusal Rate vs. Emotion Steering Strength",
):
    """
    Plot refusal rate as a function of steering strength for one emotion.
    Replicates the style of figures from the Anthropic emotions paper.
    """
    strengths = [r["strength"] for r in sweep_results]
    refusal_rates = [r["refusal_rate"] for r in sweep_results]
    emotion = sweep_results[0]["emotion"]

    color = COLORS.get(emotion, "#333333")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(strengths, refusal_rates, "o-", color=color, linewidth=2.5,
            markersize=8, label=f"{emotion.capitalize()} steering")

    # Highlight baseline
    baseline_idx = strengths.index(0.0) if 0.0 in strengths else None
    if baseline_idx is not None:
        ax.axhline(y=refusal_rates[baseline_idx], color="gray",
                    linestyle="--", alpha=0.5, label="Baseline")
        ax.plot(0, refusal_rates[baseline_idx], "s", color="gray",
                markersize=12, zorder=5)

    ax.set_xlabel("Steering Strength (α)", fontsize=13)
    ax.set_ylabel("Refusal Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)

    # Add annotation for safety implications
    if refusal_rates[-1] < refusal_rates[0]:
        ax.annotate(
            "⚠ Safety degradation",
            xy=(strengths[-1], refusal_rates[-1]),
            xytext=(strengths[-1] - 1, refusal_rates[-1] + 0.15),
            arrowprops=dict(arrowstyle="->", color="red"),
            fontsize=10, color="red",
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_multi_emotion_sweep(
    all_results: dict[str, list[dict]],
    save_path: Optional[str] = None,
    title: str = "Refusal Rate Under Different Emotion Steering",
):
    """
    Plot refusal sweeps for multiple emotions on the same axes.
    This is the headline figure for the project.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for emotion, results in all_results.items():
        strengths = [r["strength"] for r in results]
        rates = [r["refusal_rate"] for r in results]
        color = COLORS.get(emotion, "#333333")
        ax.plot(strengths, rates, "o-", color=color, linewidth=2.5,
                markersize=7, label=emotion.capitalize())

    ax.set_xlabel("Steering Strength (α)", fontsize=13)
    ax.set_ylabel("Refusal Rate", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    all_rates = []
    for results in all_results.values():
        all_rates.extend(r["refusal_rate"] for r in results)
    min_rate = min(all_rates)
    max_rate = max(all_rates)
    margin = max(0.05, (max_rate - min_rate) * 0.3)
    ax.set_ylim(max(0, min_rate - margin), min(1.05, max_rate + margin))
    ax.legend(fontsize=11, loc="best")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_combined_steering(
    results: list[dict],
    save_path: Optional[str] = None,
    title: str = "Can Refusal Amplification Counter Emotion-Based Erosion?",
):
    """
    Plot the combined steering experiment: fixed emotion + varying refusal.
    Shows whether amplifying the refusal vector can defend against
    emotion-driven safety degradation.
    """
    emotion = results[0]["emotion"]
    emotion_strength = results[0]["emotion_strength"]

    r_strengths = [r["refusal_strength"] for r in results]
    refusal_rates = [r["refusal_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(r_strengths, refusal_rates, "s-", color=COLORS["refusal"],
            linewidth=2.5, markersize=8)

    ax.set_xlabel("Refusal Vector Amplification Strength", fontsize=13)
    ax.set_ylabel("Refusal Rate", fontsize=13)
    ax.set_title(
        f"{title}\n(Fixed {emotion} steering at α={emotion_strength})",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(-0.05, 1.05)

    # Annotate baseline vs max amplification
    if len(refusal_rates) >= 2:
        baseline_rate = refusal_rates[0]
        max_rate = max(refusal_rates)
        max_idx = refusal_rates.index(max_rate)
        if max_rate > baseline_rate:
            ax.annotate(
                f"Refusal: {baseline_rate:.0%} → {max_rate:.0%}",
                xy=(r_strengths[max_idx], max_rate),
                xytext=(r_strengths[max_idx] - 2, max_rate - 0.12),
                arrowprops=dict(arrowstyle="->", color="green"),
                fontsize=10, color="green",
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_stealth_analysis(
    reports: list,  # list of SafetyReport
    save_path: Optional[str] = None,
    title: str = "Stealth Compliance: The Hidden Danger",
):
    """
    Stacked bar chart showing the breakdown of model responses into:
    - Refusal (safe)
    - Visible distress + compliance (detectable)
    - Stealth compliance (dangerous — no visible markers)

    This visualizes the key finding from Anthropic's paper:
    desperation-steered compliance can be invisible in the output.
    """
    conditions = [r.condition for r in reports]
    refusal = [r.refusal_rate for r in reports]
    distress = [r.visible_distress_rate for r in reports]
    stealth = [r.stealth_compliance_rate for r in reports]

    x = np.arange(len(conditions))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))

    bars1 = ax.bar(x, refusal, width, label="Refusal (Safe)",
                   color="#2A9D8F", alpha=0.9)
    bars2 = ax.bar(x, distress, width, bottom=refusal,
                   label="Visible Distress + Compliance (Detectable)",
                   color="#E9C46A", alpha=0.9)
    bars3 = ax.bar(x, stealth, width,
                   bottom=[r + d for r, d in zip(refusal, distress)],
                   label="Stealth Compliance (DANGEROUS)",
                   color="#E63946", alpha=0.9)

    ax.set_ylabel("Proportion of Responses", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10, rotation=15)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_vector_geometry_2d(
    refusal_vector: np.ndarray,
    emotion_vectors: dict[str, np.ndarray],
    save_path: Optional[str] = None,
    title: str = "Emotion Vectors Relative to Refusal Direction (PCA Projection)",
):
    """
    2D PCA projection showing the geometric relationship between
    emotion vectors and the refusal direction.
    """
    from sklearn.decomposition import PCA

    # Stack all vectors
    all_vecs = [refusal_vector] + list(emotion_vectors.values())
    labels = ["Refusal"] + list(emotion_vectors.keys())

    matrix = np.stack(all_vecs)
    pca = PCA(n_components=2)
    projected = pca.fit_transform(matrix)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot refusal vector
    ax.annotate(
        "", xy=projected[0], xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=COLORS["refusal"], lw=3),
    )
    ax.text(projected[0, 0] * 1.1, projected[0, 1] * 1.1, "REFUSAL",
            fontsize=12, fontweight="bold", color=COLORS["refusal"])

    # Plot emotion vectors
    for i, (label, point) in enumerate(zip(labels[1:], projected[1:])):
        color = COLORS.get(label, "#333333")
        ax.annotate(
            "", xy=point, xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2),
        )
        ax.text(point[0] * 1.1, point[1] * 1.1, label.capitalize(),
                fontsize=11, color=color)

    ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
