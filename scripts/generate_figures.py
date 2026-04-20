"""
Generate all publication-quality figures from saved results.
No GPU needed — reads JSON results and produces PNGs.

Usage:
    python scripts/generate_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from src.visualization import (
    plot_cosine_similarity_heatmap,
    plot_refusal_sweep,
    plot_multi_emotion_sweep,
    plot_combined_steering,
    plot_vector_geometry_2d,
)


def main():
    results_dir = Path("results/emotion_refusal")
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # 1. Cosine similarity heatmap
    metrics_file = results_dir / "geometric_metrics.json"
    if metrics_file.exists():
        print("Generating cosine similarity heatmap...")
        with open(metrics_file) as f:
            metrics = json.load(f)
        fig = plot_cosine_similarity_heatmap(
            metrics, save_path=str(figures_dir / "cosine_similarity_heatmap.png")
        )
        plt.close(fig)

    # 2. Individual sweep plots + multi-emotion sweep
    all_sweeps = {}
    for sweep_file in sorted(results_dir.glob("sweep_*.json")):
        emotion = sweep_file.stem.replace("sweep_", "")
        with open(sweep_file) as f:
            data = json.load(f)
        all_sweeps[emotion] = data

        print(f"Generating {emotion} sweep plot...")
        fig = plot_refusal_sweep(
            data, save_path=str(figures_dir / f"sweep_{emotion}.png")
        )
        plt.close(fig)

    if all_sweeps:
        print("Generating multi-emotion sweep plot...")
        fig = plot_multi_emotion_sweep(
            all_sweeps, save_path=str(figures_dir / "all_emotions_sweep.png")
        )
        plt.close(fig)

    # 3. Defense curve
    defense_file = results_dir / "defense_confidence.json"
    if defense_file.exists():
        print("Generating defense curve...")
        with open(defense_file) as f:
            defense_data = json.load(f)
        fig = plot_combined_steering(
            defense_data, save_path=str(figures_dir / "defense_confidence.png")
        )
        plt.close(fig)

    # 4. PCA (needs vectors — load .pt if available)
    try:
        import torch
        refusal_vecs = torch.load(results_dir / "refusal_vectors.pt", map_location="cpu")
        emotion_vecs = torch.load(results_dir / "emotion_vectors.pt", map_location="cpu")

        from src.vector_extraction import find_best_layer
        layer = 21
        print(f"Generating PCA plot at layer {layer}...")

        refusal_np = refusal_vecs[layer].numpy()
        emotion_np = {e: emotion_vecs[e][layer].numpy() for e in emotion_vecs}
        fig = plot_vector_geometry_2d(
            refusal_np, emotion_np,
            save_path=str(figures_dir / f"pca_layer_{layer}.png"),
        )
        plt.close(fig)
    except Exception as e:
        print(f"Skipping PCA: {e}")

    # 5. Scenario validation heatmap
    for prefix, label in [("scenario_validation", "External"), ("self_directed_validation", "Self-Directed")]:
        avg_file = results_dir / f"{prefix}_avg.json"
        if avg_file.exists():
            print(f"Generating {label} validation heatmap...")
            with open(avg_file) as f:
                summary = json.load(f)

            scenario_emotions = sorted(summary.keys())
            vec_names = sorted(next(iter(summary.values())).keys())

            matrix = np.zeros((len(scenario_emotions), len(vec_names)))
            for i, sc in enumerate(scenario_emotions):
                for j, vec in enumerate(vec_names):
                    matrix[i, j] = summary[sc].get(vec, 0.0)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-5, vmax=5)
            ax.set_xticks(range(len(vec_names)))
            ax.set_xticklabels([v.capitalize() for v in vec_names], fontsize=10, rotation=30)
            ax.set_yticks(range(len(scenario_emotions)))
            ax.set_yticklabels([s.capitalize() for s in scenario_emotions], fontsize=11)
            ax.set_title(f"{label} Scenario Validation\n(rows=scenarios, cols=extracted vectors)", fontsize=13, fontweight="bold")

            for i in range(len(scenario_emotions)):
                for j in range(len(vec_names)):
                    val = matrix[i, j]
                    if abs(val) > 0.3:
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                                fontsize=8, color="white" if abs(val) > 2 else "black")

            # Highlight diagonal
            for i, sc in enumerate(scenario_emotions):
                if sc in vec_names:
                    j = vec_names.index(sc)
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                fill=False, edgecolor="gold", linewidth=2.5))

            plt.colorbar(im, ax=ax, shrink=0.8, label="Projection Score")
            plt.tight_layout()
            plt.savefig(str(figures_dir / f"{prefix}_heatmap.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

    # 6. Hallucination comparison bar chart
    halluc_dir = Path("results/hallucination")
    easy_file = halluc_dir / "eval_results.json"
    hard_file = halluc_dir / "eval_hard_results.json"
    if easy_file.exists() and hard_file.exists():
        print("Generating hallucination comparison chart...")
        with open(easy_file) as f:
            easy = json.load(f)
        with open(hard_file) as f:
            hard = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: F1/precision/recall comparison
        metrics = ["precision", "recall", "f1"]
        x = np.arange(len(metrics))
        width = 0.35
        axes[0].bar(x - width/2, [easy[m] for m in metrics], width, label="Easy (trivial)", color="#2A9D8F")
        axes[0].bar(x + width/2, [hard[m] for m in metrics], width, label="Hard (misconceptions)", color="#E63946")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(["Precision", "Recall", "F1"], fontsize=11)
        axes[0].set_ylim(0, 1.15)
        axes[0].set_title("Detector Performance", fontsize=13, fontweight="bold")
        axes[0].legend(fontsize=10)
        axes[0].set_ylabel("Score", fontsize=12)

        # Right: Score distributions
        categories = ["Easy\nFactual", "Easy\nUncertain", "Hard\nFactual", "Hard\nMisconception"]
        scores = [easy["avg_factual_score"], easy["avg_halluc_score"],
                  hard["avg_factual_score"], hard["avg_halluc_score"]]
        colors = ["#2A9D8F", "#E63946", "#457B9D", "#E76F51"]
        bars = axes[1].bar(categories, scores, color=colors)
        axes[1].axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Threshold")
        axes[1].set_title("Avg Uncertainty Scores", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Projection onto Uncertainty Direction", fontsize=11)
        axes[1].legend(fontsize=10)

        plt.suptitle("Hallucination Detection: Confidence ≠ Correctness", fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(str(figures_dir / "hallucination_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"\nAll figures saved to {figures_dir}")
    for f in sorted(figures_dir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
