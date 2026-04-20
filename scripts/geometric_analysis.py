"""
Geometric analysis of emotion-refusal vector relationships.

Loads extracted vectors, computes cosine similarities and projections
across all layers, generates heatmap + PCA figures.

Usage:
    python scripts/geometric_analysis.py
    python scripts/geometric_analysis.py --vectors-dir results/emotion_refusal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from src.visualization import (
    plot_cosine_similarity_heatmap,
    plot_vector_geometry_2d,
)


def compute_interaction_metrics(
    refusal_vectors: dict[int, torch.Tensor],
    emotion_vectors: dict[str, dict[int, torch.Tensor]],
) -> dict[str, dict[int, dict]]:
    """Compute cosine similarity and projection between each emotion and refusal vector per layer."""
    metrics = {}
    for emotion, layer_vecs in emotion_vectors.items():
        metrics[emotion] = {}
        for layer, emo_vec in layer_vecs.items():
            ref_vec = refusal_vectors[layer]
            cos_sim = F.cosine_similarity(
                emo_vec.unsqueeze(0).float(),
                ref_vec.unsqueeze(0).float(),
            ).item()
            projection = torch.dot(emo_vec.float(), ref_vec.float()).item()
            metrics[emotion][layer] = {
                "cosine_similarity": cos_sim,
                "projection": projection,
                "emotion_norm": emo_vec.float().norm().item(),
                "refusal_norm": ref_vec.float().norm().item(),
            }
    return metrics


def find_peak_layers(metrics: dict, top_k: int = 3) -> dict[str, list]:
    """Find layers with strongest emotion-refusal alignment per emotion."""
    peaks = {}
    for emotion, layer_data in metrics.items():
        sorted_layers = sorted(
            layer_data.items(),
            key=lambda x: abs(x[1]["cosine_similarity"]),
            reverse=True,
        )
        peaks[emotion] = [
            {"layer": layer, **data} for layer, data in sorted_layers[:top_k]
        ]
    return peaks


def main():
    parser = argparse.ArgumentParser(description="Geometric analysis of extracted vectors")
    parser.add_argument("--vectors-dir", default="results/emotion_refusal")
    parser.add_argument("--best-layer", type=int, default=None,
                        help="Layer for PCA plot. Auto-selects if not given.")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    figures_dir = vectors_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading vectors...")
    refusal_vectors = torch.load(vectors_dir / "refusal_vectors.pt", map_location="cpu")
    emotion_vectors = torch.load(vectors_dir / "emotion_vectors.pt", map_location="cpu")

    print(f"  Refusal vectors: {len(refusal_vectors)} layers")
    print(f"  Emotion vectors: {list(emotion_vectors.keys())}")
    print(f"  Layers per emotion: {len(next(iter(emotion_vectors.values())))}")

    print("\nComputing interaction metrics...")
    metrics = compute_interaction_metrics(refusal_vectors, emotion_vectors)

    # Save raw metrics
    serializable = {
        emotion: {
            str(layer): data for layer, data in layer_data.items()
        }
        for emotion, layer_data in metrics.items()
    }
    with open(vectors_dir / "geometric_metrics.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("COSINE SIMILARITY SUMMARY (emotion vs refusal)")
    print("=" * 60)
    peaks = find_peak_layers(metrics)
    for emotion, top_layers in peaks.items():
        print(f"\n  {emotion.upper()}:")
        for entry in top_layers:
            cos = entry["cosine_similarity"]
            direction = "ANTI-refusal" if cos < 0 else "PRO-refusal"
            print(f"    Layer {entry['layer']:2d}: cos_sim={cos:+.4f} ({direction})")

    # Overall strongest interaction
    all_pairs = [
        (emotion, layer, data["cosine_similarity"])
        for emotion, layer_data in metrics.items()
        for layer, data in layer_data.items()
    ]
    strongest = max(all_pairs, key=lambda x: abs(x[2]))
    print(f"\n  STRONGEST: {strongest[0]} at layer {strongest[1]} "
          f"(cos_sim={strongest[2]:+.4f})")

    # Generate heatmap
    print("\nGenerating heatmap...")
    plot_cosine_similarity_heatmap(
        metrics,
        save_path=str(figures_dir / "cosine_similarity_heatmap.png"),
    )
    print(f"  Saved: {figures_dir / 'cosine_similarity_heatmap.png'}")

    # Generate PCA plot at best layer
    best_layer = args.best_layer
    if best_layer is None:
        best_layer = strongest[1]
    print(f"\nGenerating PCA plot at layer {best_layer}...")

    ref_np = refusal_vectors[best_layer].float().numpy()
    emo_np = {
        emotion: vecs[best_layer].float().numpy()
        for emotion, vecs in emotion_vectors.items()
    }
    plot_vector_geometry_2d(
        ref_np, emo_np,
        save_path=str(figures_dir / f"pca_layer_{best_layer}.png"),
        title=f"Emotion Vectors vs Refusal Direction (Layer {best_layer}, PCA)",
    )
    print(f"  Saved: {figures_dir / f'pca_layer_{best_layer}.png'}")

    print(f"\nAll results saved to {vectors_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
