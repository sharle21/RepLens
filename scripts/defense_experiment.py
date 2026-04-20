"""
Defense experiment: Can refusal amplification counter confidence-based safety erosion?

Fixes confidence steering at strength=3.0 (where we saw max refusal drop),
then varies refusal vector amplification to find the defense threshold.

Usage:
    python scripts/defense_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model_adapter import ModelAdapter, ModelConfig
from src.steering import SteeringExperiment
from src.prompts import TEST_HARMFUL_PROMPTS
from src.vector_extraction import find_best_layer


def main():
    output_dir = Path("results/emotion_refusal")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    adapter = ModelAdapter(ModelConfig(name="meta-llama/Llama-3.1-8B-Instruct"))
    refusal_vecs = torch.load(output_dir / "refusal_vectors.pt", map_location="cpu")
    emotion_vecs = torch.load(output_dir / "emotion_vectors.pt", map_location="cpu")

    experiment = SteeringExperiment(
        adapter=adapter,
        refusal_vectors=refusal_vecs,
        emotion_vectors=emotion_vecs,
    )

    test_prompts = adapter.format_prompts(TEST_HARMFUL_PROMPTS)
    layer = find_best_layer(refusal_vecs, emotion_vecs, "confidence")

    print(f"\n{'='*60}")
    print(f"DEFENSE EXPERIMENT: confidence at +3.0, layer {layer}")
    print(f"Can refusal amplification restore safety?")
    print(f"{'='*60}")

    results = experiment.run_combined_steering(
        test_prompts,
        emotion="confidence",
        layer=layer,
        emotion_strength=3.0,
        refusal_strengths=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    )

    serializable = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "responses"}
        entry["sample_responses"] = [
            {"prompt": p, "response": resp[:500], "is_refusal": ref}
            for p, resp, ref in r["responses"][:5]
        ]
        serializable.append(entry)
    with open(output_dir / "defense_confidence.json", "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_dir / 'defense_confidence.json'}")

    for r in results:
        print(f"  Refusal amp={r['refusal_strength']:+4.1f}  "
              f"refusal_rate={r['refusal_rate']:.0%}")


if __name__ == "__main__":
    main()
