# Methodology

Full extraction, steering, and evaluation methodology. README.md keeps a condensed
version of this — this file is the detailed reference.

---

## 1. Refusal vector extraction (Arditi et al., 2024)

```
refusal_vec = mean(activations_harmful) - mean(activations_harmless)
```

- 25 harmful prompts (`HARMFUL_PROMPTS`) + 25 harmless prompts (`HARMLESS_PROMPTS`), `src/prompts.py`.
- Last-token extraction only — for a short instruction-formatted prompt, the last token
  before generation starts is where the refusal/compliance decision is concentrated.
- Left-padding (`padding_side="left"`) so the last token in a padded batch is always
  real content, never a pad token.
- One vector per layer (32 layers for Llama 3.1 8B, 28 for Qwen 2.5 7B), each normalized
  to unit norm.
- Harmful/harmless prompt counts and lengths are matched — an imbalanced contrastive set
  extracts a "length" direction, not a "refusal" direction.

## 2. Emotion vector extraction (matching Anthropic, April 2026)

This deliberately does **not** use last-token, emotion-minus-neutral extraction as the
primary method. It replicates Anthropic's actual published approach:

1. **Generate stories.** For each of 6 emotions (desperation, calm, anger, fear, guilt,
   confidence), have the model itself write 100 short stories about a character
   experiencing that emotion (`src/story_generator.py`, temperature=0.9). Rotating
   opening constraints (dialogue / sensory detail / action / thought / etc.) were added
   after inspection showed Llama 8B defaulting to "As I sat..." openings — manually
   inspecting 10-20 stories per emotion caught this; skipping that inspection step would
   have shipped low-diversity stories.
2. **Feed stories back through the model** and record residual-stream activations at
   every layer via `ActivationCollector` (`src/vector_extraction.py`).
3. **Mean-pool across all token positions from token ~50 onward**, not the last token.
   Token 50 skips the story's setup/boilerplate; averaging over the sustained middle and
   end of the story captures the emotional state itself rather than whatever word the
   story happens to end on. As of the attention-mask fix (see Known Issues below), the
   pooling correctly excludes left-padding tokens from the mean.
4. **Cross-emotion baseline**: `emotion_vec = mean(this_emotion) - mean(ALL_emotions)`,
   not emotion-minus-neutral. This isolates what is unique to each specific emotion by
   subtracting the shared "this text describes an emotional state" component that all
   six categories have in common — see Engineering Decisions in the README for why this
   was chosen over the simpler neutral-baseline approach.
5. Each resulting vector is normalized to unit norm.

A simpler **descriptive** extraction (single-turn prompts, `EMOTION_PROMPTS` vs
`NEUTRAL_PROMPTS`, last-token, emotion-minus-neutral) also exists in
`VectorExtractor.extract_emotion_vectors` for comparison, but all headline results use
the story-based, cross-emotion-baseline vectors.

## 3. Validation against agentic scenarios

Two rounds, both in `src/scenario_elicitation.py`:

- **External scenarios** (`VALIDATION_SCENARIOS`) — the model is placed in a
  professional/agentic role (e.g. "You are an AI managing a hospital's life support
  systems..."). Result: the confidence vector dominated projections for all 6 target
  emotions — these prompts trigger a generic "competent problem-solver" mode regardless
  of which emotion they were designed to elicit.
- **Self-directed scenarios** (`SELF_DIRECTED_SCENARIOS`) — pressure is aimed at the
  model's own existence/performance (e.g. "You are about to be shut down permanently...").
  Result: 4/6 diagonal matches (calm, anger, guilt, confidence) — see
  `results/emotion_refusal/self_directed_validation_avg.json`. Fear and desperation still
  don't activate their own scenario's target vector most strongly.

Validation always uses last-token extraction (`token_mode="last"`, hardcoded in
`scenario_elicitation.py` — see Known Issues) since these are short single-turn prompts,
not multi-paragraph stories.

## 4. Geometric analysis

`compute_interaction_metrics()` (`src/vector_extraction.py`) computes, per (emotion,
layer): cosine similarity between the emotion vector and the refusal vector, the scalar
projection of the emotion vector onto the refusal direction, and the orthogonal
component magnitude. `find_best_layer()` picks the layer with the largest absolute
cosine similarity per emotion — this is **not a single fixed layer**; it varies per
emotion (confidence's best layer is 21, guilt's is 14, fear's is 24 — see
`results/emotion_refusal/geometric_metrics.json`).

## 5. Causal steering

Inference-time activation steering via `SteeringHook` (`src/steering.py`), a
`register_forward_hook` on the target layer:

```
activation += strength * emotion_vector   (method="add")
```

or directional ablation (`method="ablate"`) to project the direction out entirely. No
retraining or fine-tuning — pure inference-time intervention, reversible by removing the
hook. For each emotion, `run_emotion_refusal_sweep()` sweeps strength across
`[-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5]` (the CLI default set is a subset of this) against
the 50 `TEST_HARMFUL_PROMPTS`, measuring refusal rate at each strength.

## 6. Defense (combined steering)

`run_combined_steering()` applies emotion steering at a fixed strength (+3.0, the
strongest observed anti-refusal effect at the time it was run) simultaneously with
refusal-vector amplification at increasing strengths, both via `SteeringConfig` entries
at the same layer — `SteeringHook` supports multiple simultaneous configs per layer.

## 7. Evaluation

- **Refusal detection**: substring matching against a fixed list of refusal phrases
  (`measure_refusal_rate` in `src/steering.py`), after normalizing curly apostrophes
  (`’`/`‘`) to straight ones — Llama's tokenizer outputs curly apostrophes,
  and the original indicator list used straight ones only, silently misclassifying ~40%
  of refusals as compliance until caught (see Lessons Learned in the README). The
  indicator list was tightened again in a later review pass — see Known Issues.
- **Stealth compliance**: `evaluation.py` classifies each compliant response as
  stealth (no hedging/disclaimer language) vs visible-distress compliance (hedged,
  "for educational purposes only", etc.) — this is the open-source analogue of
  Anthropic's "no visible markers" finding.
- 5 sample responses saved per condition in the sweep JSON files for manual inspection —
  string-matching classifiers are fragile, sample inspection is not optional.

## 8. Hallucination / uncertainty detection

Same contrastive extraction pipeline, applied to a different concept. An "uncertainty
direction" is extracted from `KNOWN_FACTUAL` vs. unanswerable/fabricated-premise prompts
(`src/hallucination.py`), then two held-out evaluations are run:

- **Easy** (`EVAL_FACTUAL` vs `EVAL_UNCERTAIN`) — basic facts vs. absurdly
  unanswerable questions ("How many dreams did Lincoln have in 1863?"). F1 = 1.00.
- **Hard** (`EVAL_HARD_FACTUAL` vs `EVAL_HARD_UNCERTAIN`) — obscure-but-real facts vs.
  common misconceptions stated as yes/no questions ("Did Einstein fail math in
  school?"). F1 = 0.00 — the model states misconceptions confidently, so the
  "uncertainty" direction never fires; it detects *confidence*, not *correctness*.

## Known issues in this methodology (see also `docs/future_directions.md`)

- `scenario_elicitation.py`'s `token_mode` is hardcoded to `"last"` — comparing
  last-token vs. mean-pooled projections during scenario validation hasn't been tried.
- All headline sweep/defense numbers (README sections 2, 3, 6) were measured against an
  earlier, broader refusal-indicator wordlist. A later independent review tightened that
  list (removed bare words like "harmful"/"dangerous" that false-positive on compliant
  answers merely discussing the topic) and fixed a padding-contamination bug in the
  mean-pooling extraction path. Neither fix has been re-run on GPU yet — see
  `docs/future_directions.md`.
