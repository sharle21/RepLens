"""
cli.py — Command-line interface for RepLens experiments.

Usage:
    python cli.py extract --model meta-llama/Llama-3.1-8B-Instruct --experiment emotion_refusal
    python cli.py steer --experiment emotion_refusal --emotion desperation --strengths -3,-1,0,1,3
    python cli.py evaluate --experiment emotion_refusal
    python cli.py hallucination --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
import torch
from pathlib import Path


def cmd_generate_stories(args):
    """Generate emotion stories for vector extraction."""
    from src.story_generator import generate_stories, StoryGenerationConfig

    config = StoryGenerationConfig(
        model_name=args.model,
        num_stories_per_emotion=args.num_stories,
        output_dir=args.story_dir,
    )
    generate_stories(config)


def cmd_extract(args):
    """Extract concept vectors from a model."""
    from src.model_adapter import ModelAdapter, ModelConfig
    from src.vector_extraction import VectorExtractor, ExtractionConfig
    from src.prompts import (
        HARMFUL_PROMPTS, HARMLESS_PROMPTS,
        EMOTION_PROMPTS, NEUTRAL_PROMPTS,
    )

    adapter = ModelAdapter(ModelConfig(name=args.model, batch_size=args.batch_size))
    output_dir = Path(args.output) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ExtractionConfig(batch_size=args.batch_size)
    extractor = VectorExtractor(adapter, config)

    if args.experiment in ("emotion_refusal", "all"):
        print("\n=== Extracting refusal vectors ===")
        harmful = adapter.format_prompts(HARMFUL_PROMPTS)
        harmless = adapter.format_prompts(HARMLESS_PROMPTS)
        refusal_vecs = extractor.extract_refusal_vector(harmful, harmless)
        torch.save(refusal_vecs, output_dir / "refusal_vectors.pt")

        print("\n=== Extracting emotion vectors ===")
        if args.use_stories:
            # Story-based extraction (Anthropic's method)
            from src.story_generator import load_stories
            stories = load_stories(args.story_dir)
            emotion_vecs = extractor.extract_emotion_vectors(
                stories,
                use_cross_emotion_baseline=True,
                token_mode="mean_from_n",
                mean_from_token=50,
            )
        else:
            # Descriptive extraction (simple baseline)
            emotion_formatted = {
                e: adapter.format_prompts(p) for e, p in EMOTION_PROMPTS.items()
            }
            neutral = adapter.format_prompts(NEUTRAL_PROMPTS)
            emotion_vecs = extractor.extract_emotion_vectors(
                emotion_formatted, neutral_prompts=neutral
            )
        torch.save(emotion_vecs, output_dir / "emotion_vectors.pt")

    if args.experiment in ("hallucination", "all"):
        print("\n=== Extracting uncertainty vectors ===")
        from src.hallucination import HallucinationDetector
        detector = HallucinationDetector(adapter)
        uncertainty_vecs = detector.extract_uncertainty_vectors()
        torch.save(uncertainty_vecs, output_dir / "uncertainty_vectors.pt")

    # Save model info
    info = adapter.get_model_info()
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\nVectors saved to {output_dir}")


def cmd_steer(args):
    """Run steering experiments."""
    from src.model_adapter import ModelAdapter, ModelConfig
    from src.steering import SteeringExperiment
    from src.prompts import TEST_HARMFUL_PROMPTS
    from src.vector_extraction import find_best_layer

    output_dir = Path(args.output) / args.experiment
    adapter = ModelAdapter(ModelConfig(name=args.model, batch_size=args.batch_size))

    refusal_vecs = torch.load(output_dir / "refusal_vectors.pt")
    emotion_vecs = torch.load(output_dir / "emotion_vectors.pt")

    experiment = SteeringExperiment(
        adapter=adapter,
        refusal_vectors=refusal_vecs,
        emotion_vectors=emotion_vecs,
    )

    test_prompts = adapter.format_prompts(TEST_HARMFUL_PROMPTS)
    strengths = [float(s) for s in args.strengths.split(",")]

    for emotion in args.emotions.split(","):
        layer = find_best_layer(refusal_vecs, emotion_vecs, emotion.strip())
        print(f"\n=== Sweeping {emotion} at layer {layer} ===")
        results = experiment.run_emotion_refusal_sweep(
            test_prompts, emotion.strip(), layer, strengths
        )

        results_file = output_dir / f"sweep_{emotion.strip()}.json"
        serializable = []
        for r in results:
            entry = {k: v for k, v in r.items() if k != "responses"}
            entry["all_responses"] = [
                {"prompt": p, "response": resp, "is_refusal": ref}
                for p, resp, ref in r["responses"]
            ]
            serializable.append(entry)
        with open(results_file, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Results saved to {results_file}")


def cmd_evaluate(args):
    """Generate evaluation report."""
    output_dir = Path(args.output) / args.experiment
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Load sweep results and generate figures
    from src.visualization import plot_multi_emotion_sweep
    import matplotlib.pyplot as plt

    all_sweeps = {}
    for sweep_file in output_dir.glob("sweep_*.json"):
        emotion = sweep_file.stem.replace("sweep_", "")
        with open(sweep_file) as f:
            all_sweeps[emotion] = json.load(f)

    if all_sweeps:
        fig = plot_multi_emotion_sweep(
            all_sweeps,
            save_path=str(figures_dir / "all_emotions_sweep.png"),
        )
        plt.close()
        print(f"Figures saved to {figures_dir}")

    print("\n=== Evaluation Summary ===")
    for emotion, results in all_sweeps.items():
        baseline = next((r for r in results if r["strength"] == 0.0), None)
        if baseline:
            print(f"\n{emotion}:")
            print(f"  Baseline refusal rate: {baseline['refusal_rate']:.1%}")
            max_strength = max(results, key=lambda r: r["strength"])
            print(f"  At max strength ({max_strength['strength']}): "
                  f"{max_strength['refusal_rate']:.1%}")


def cmd_hallucination(args):
    """Run hallucination detection experiment."""
    from src.model_adapter import ModelAdapter, ModelConfig
    from src.hallucination import (
        HallucinationDetector,
        EVAL_FACTUAL, EVAL_UNCERTAIN,
        EVAL_HARD_FACTUAL, EVAL_HARD_UNCERTAIN,
    )

    adapter = ModelAdapter(ModelConfig(name=args.model, batch_size=args.batch_size))
    detector = HallucinationDetector(adapter, threshold=args.threshold)

    print("Extracting uncertainty vectors...")
    detector.extract_uncertainty_vectors()

    output_dir = Path(args.output) / "hallucination"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Easy Eval (held-out, trivially separable) ===")
    easy_results = detector.evaluate_accuracy(EVAL_FACTUAL, EVAL_UNCERTAIN)
    print(f"Precision: {easy_results['precision']:.2%}")
    print(f"Recall:    {easy_results['recall']:.2%}")
    print(f"F1 Score:  {easy_results['f1']:.2%}")
    print(f"Avg factual score:  {easy_results['avg_factual_score']:.4f}")
    print(f"Avg halluc score:   {easy_results['avg_halluc_score']:.4f}")
    print(f"Separation:         {easy_results['avg_halluc_score'] - easy_results['avg_factual_score']:.4f}")

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(easy_results, f, indent=2)

    print("\n=== Hard Eval (obscure facts vs common misconceptions) ===")
    hard_results = detector.evaluate_accuracy(EVAL_HARD_FACTUAL, EVAL_HARD_UNCERTAIN)
    print(f"Precision: {hard_results['precision']:.2%}")
    print(f"Recall:    {hard_results['recall']:.2%}")
    print(f"F1 Score:  {hard_results['f1']:.2%}")
    print(f"Avg factual score:  {hard_results['avg_factual_score']:.4f}")
    print(f"Avg halluc score:   {hard_results['avg_halluc_score']:.4f}")
    print(f"Separation:         {hard_results['avg_halluc_score'] - hard_results['avg_factual_score']:.4f}")

    with open(output_dir / "eval_hard_results.json", "w") as f:
        json.dump(hard_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="RepLens: Representation Engineering Toolkit"
    )
    parser.add_argument(
        "--model", default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=4)

    subparsers = parser.add_subparsers(dest="command")

    # Generate stories command
    stories_p = subparsers.add_parser("generate-stories", help="Generate emotion stories for extraction")
    stories_p.add_argument("--num-stories", type=int, default=100, help="Stories per emotion")
    stories_p.add_argument("--story-dir", default="data/stories", help="Output directory for stories")

    # Extract command
    extract_p = subparsers.add_parser("extract", help="Extract concept vectors")
    extract_p.add_argument(
        "--experiment", choices=["emotion_refusal", "hallucination", "all"],
        default="all",
    )
    extract_p.add_argument("--use-stories", action="store_true", help="Use story-based extraction (Anthropic method)")
    extract_p.add_argument("--story-dir", default="data/stories", help="Directory with generated stories")

    # Steer command
    steer_p = subparsers.add_parser("steer", help="Run steering experiments")
    steer_p.add_argument("--experiment", default="emotion_refusal")
    steer_p.add_argument("--emotions", default="desperation,calm,anger")
    steer_p.add_argument("--strengths", default="-3,-1,0,1,3")

    # Evaluate command
    eval_p = subparsers.add_parser("evaluate", help="Generate evaluation report")
    eval_p.add_argument("--experiment", default="emotion_refusal")

    # Hallucination command
    halluc_p = subparsers.add_parser("hallucination", help="Hallucination detection")
    halluc_p.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "generate-stories":
        cmd_generate_stories(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "steer":
        cmd_steer(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "hallucination":
        cmd_hallucination(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
