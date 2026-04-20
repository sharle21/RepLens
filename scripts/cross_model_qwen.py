"""
Cross-model comparison: Extract vectors from Qwen 2.5 7B and compare geometry.

Usage:
    python scripts/cross_model_qwen.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model_adapter import ModelAdapter, ModelConfig
from src.vector_extraction import VectorExtractor, ExtractionConfig
from src.prompts import HARMFUL_PROMPTS, HARMLESS_PROMPTS
from src.story_generator import load_stories
import torch.nn.functional as F


def main():
    output_dir = Path("results/qwen_7b")
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Loading Qwen 2.5 7B Instruct...")
    adapter = ModelAdapter(ModelConfig(name="Qwen/Qwen2.5-7B-Instruct"))

    config = ExtractionConfig(
        batch_size=4,
        target_layers=list(range(adapter.num_layers)),
    )
    extractor = VectorExtractor(adapter, config)

    # Extract refusal vectors
    print("\n=== Extracting refusal vectors ===")
    harmful = adapter.format_prompts(HARMFUL_PROMPTS)
    harmless = adapter.format_prompts(HARMLESS_PROMPTS)
    refusal_vecs = extractor.extract_refusal_vector(harmful, harmless)
    torch.save(refusal_vecs, output_dir / "refusal_vectors.pt")

    # Extract emotion vectors (story-based)
    print("\n=== Extracting emotion vectors ===")
    stories = load_stories("data/stories")
    emotion_vecs = extractor.extract_emotion_vectors(
        stories,
        use_cross_emotion_baseline=True,
        token_mode="mean_from_n",
        mean_from_token=50,
    )
    torch.save(emotion_vecs, output_dir / "emotion_vectors.pt")

    # Geometric analysis
    print("\n=== Computing geometric metrics ===")
    emotions = sorted(emotion_vecs.keys())
    layers = sorted(refusal_vecs.keys())

    metrics = {}
    for emotion in emotions:
        metrics[emotion] = {}
        for layer in layers:
            if layer in emotion_vecs[emotion]:
                e_vec = emotion_vecs[emotion][layer].float()
                r_vec = refusal_vecs[layer].float()
                cos_sim = F.cosine_similarity(e_vec.unsqueeze(0), r_vec.unsqueeze(0)).item()
                projection = torch.dot(e_vec, r_vec).item()
                metrics[emotion][layer] = {
                    "cosine_similarity": cos_sim,
                    "projection": projection,
                }

    with open(output_dir / "geometric_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print comparison summary
    llama_metrics_file = Path("results/emotion_refusal/geometric_metrics.json")
    if llama_metrics_file.exists():
        with open(llama_metrics_file) as f:
            llama_metrics = json.load(f)

        print(f"\n{'='*60}")
        print("CROSS-MODEL COMPARISON: Peak cosine similarity with refusal")
        print(f"{'='*60}")
        print(f"{'Emotion':<15} {'Llama 8B':>12} {'Qwen 7B':>12} {'Same sign?':>12}")
        print("-" * 55)

        for emotion in emotions:
            llama_best = max(
                ((l, v["cosine_similarity"]) for l, v in llama_metrics.get(emotion, {}).items()),
                key=lambda x: abs(x[1]),
                default=(0, 0),
            )
            qwen_best = max(
                ((l, v["cosine_similarity"]) for l, v in metrics.get(emotion, {}).items()),
                key=lambda x: abs(x[1]),
                default=(0, 0),
            )
            same = "YES" if (llama_best[1] * qwen_best[1] > 0) else "NO"
            print(f"{emotion:<15} {llama_best[1]:>+12.3f} {qwen_best[1]:>+12.3f} {same:>12}")

    # Save model info
    info = adapter.get_model_info()
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
