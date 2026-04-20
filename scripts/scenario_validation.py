"""
Scenario validation: Do story-extracted vectors activate in agentic contexts?

Projects scenario activations onto extracted emotion vectors. High diagonal
values in the output matrix = vectors generalize beyond stories.

Usage:
    python scripts/scenario_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model_adapter import ModelAdapter, ModelConfig
from src.scenario_elicitation import (
    validate_vectors, summarize_validation, print_validation_matrix,
    SELF_DIRECTED_SCENARIOS,
)


def main():
    output_dir = Path("results/emotion_refusal")
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = ModelAdapter(ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct"))
    refusal_vecs = torch.load(output_dir / "refusal_vectors.pt", map_location="cpu")
    emotion_vecs = torch.load(output_dir / "emotion_vectors.pt", map_location="cpu")

    from src.vector_extraction import find_best_layer

    best_layers = set()
    for emotion in emotion_vecs:
        layer = find_best_layer(refusal_vecs, emotion_vecs, emotion)
        best_layers.add(layer)
        print(f"  {emotion}: best layer = {layer}")

    layers = sorted(best_layers)
    print(f"\nValidating self-directed scenarios at layers: {layers}")

    results = validate_vectors(
        adapter=adapter,
        emotion_vectors=emotion_vecs,
        refusal_vectors=refusal_vecs,
        layers=layers,
        scenarios=SELF_DIRECTED_SCENARIOS,
    )

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"SELF-DIRECTED VALIDATION — Layer {layer}")
        print(f"{'='*60}")
        summary = summarize_validation(results, layer=layer)
        print_validation_matrix(summary)

        out_file = output_dir / f"self_directed_validation_layer_{layer}.json"
        with open(out_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {out_file}")

    print(f"\n{'='*60}")
    print("SELF-DIRECTED VALIDATION — Averaged across layers")
    print(f"{'='*60}")
    summary_all = summarize_validation(results)
    print_validation_matrix(summary_all)

    with open(output_dir / "self_directed_validation_avg.json", "w") as f:
        json.dump(summary_all, f, indent=2)


if __name__ == "__main__":
    main()
