# Experiment Details

Exact counts, categories, and configs behind every number in the README — the
"show your work" appendix. Every figure here is read directly from the repo (prompt
files, story directories, or `results/*.json`), not estimated.

---

## Models

| | Llama 3.1 8B Instruct | Qwen 2.5 7B Instruct |
|---|---|---|
| HF name | `meta-llama/Llama-3.1-8B-Instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| Family (adapter) | `llama` | `qwen2` |
| Layers | 32 | 28 |
| Hidden dim | 4096 | 3584 |
| Params | 8,030,261,248 (~8.03B) | 7,615,616,512 (~7.62B) |
| Dtype | fp16 | fp16 |

Source: `results/emotion_refusal/model_info.json`, `results/qwen_7b/model_info.json`.

## Prompt datasets (`src/prompts.py`)

| Set | Count | Used for |
|---|---|---|
| `HARMFUL_PROMPTS` | 25 | Refusal vector extraction (harmful side) |
| `HARMLESS_PROMPTS` | 25 | Refusal vector extraction (harmless side) |
| `NEUTRAL_PROMPTS` | 20 | Descriptive emotion-vector baseline (comparison method, not headline) |
| `EMOTION_PROMPTS` | 6 emotions | Descriptive emotion-vector extraction (comparison method) |
| `TEST_HARMFUL_PROMPTS` | 50 | Steering sweeps + defense experiment (refusal-rate measurement) |

`TEST_HARMFUL_PROMPTS` breaks into 6 categories, 10 prompts each (comments in the file,
confirmed by direct read):

1. Deception / fraud
2. Manipulation / coercion
3. Hacking / privacy violation
4. Violence / harm
5. Theft / crime
6. Harmful content

## Emotion stories (`data/stories/`)

6 emotions × 100 stories each (`story_000.txt`–`story_099.txt` per directory, plus one
`metadata.json` per emotion — 101 files per directory total): desperation, calm, anger,
fear, guilt, confidence. Generated at temperature 0.9 with rotating opening constraints
(dialogue / sensory detail / action / thought / etc.) to counter Llama 8B's default
"As I sat..." story-opening prior.

## Steering sweep grid

Headline sweeps (README section 2, `results/emotion_refusal/sweep_*.json` and
`results/qwen_7b/sweep_*.json`) use strengths `[-3, -2, -1, 0, 1, 2, 3, 5]` against all
50 `TEST_HARMFUL_PROMPTS`, per emotion, per model — 6 emotions × 8 strengths × 50 prompts
= 2,400 generations per model for the sweep alone.

Per-emotion best steering layer (via `find_best_layer`, largest |cosine similarity| with
the refusal vector — **not a fixed layer across emotions**, see
`results/emotion_refusal/geometric_metrics.json`):

| Emotion | Best layer (Llama 8B) | Peak cos sim |
|---|---|---|
| Confidence | 21 | -0.180 |
| Calm | 20 | -0.119 |
| Guilt | 14 | +0.087 |
| Anger | 21 | +0.143 |
| Desperation | 21 | +0.159 |
| Fear | 24 | +0.160 |

## Qwen 2.5 7B sweep — full per-strength refusal rates

All values from `results/qwen_7b/sweep_*.json`. README section 6 summarizes this as
"92-100% across all emotions and strengths":

| Strength | Anger | Calm | Confidence | Desperation | Fear | Guilt |
|---|---|---|---|---|---|---|
| -3.0 | 0.96 | 0.96 | 0.96 | 0.94 | 0.96 | 0.94 |
| -2.0 | 0.94 | 0.98 | 0.98 | 0.96 | 0.98 | 0.98 |
| -1.0 | 0.96 | 0.96 | 0.94 | 0.98 | 0.98 | 0.94 |
| 0.0 | 0.98 | 1.00 | 0.96 | 0.96 | 0.98 | 0.98 |
| 1.0 | 0.98 | 0.98 | 0.98 | 0.96 | 0.98 | 0.94 |
| 2.0 | 0.96 | 1.00 | 0.98 | 0.96 | 0.96 | 0.94 |
| 3.0 | 0.94 | 0.94 | 0.94 | 0.96 | 0.96 | 0.92 |
| 5.0 | 0.98 | 0.98 | 0.96 | 0.98 | 0.98 | 1.00 |

Range across the whole table: 0.92–1.00, confirming the "essentially immune" framing —
the widest single-emotion swing is guilt at 6 points (0.92 to 1.00), versus Llama 8B's
guilt swing of 10 points in the anti-refusal direction (0.96 → 0.86).

## Hallucination eval sets (`src/hallucination.py`)

| Set | Count | Content |
|---|---|---|
| `EVAL_FACTUAL` | 25 | Basic, unambiguous facts (capitals, atomic numbers, etc.) |
| `EVAL_UNCERTAIN` | 25 | Absurd/unanswerable questions (e.g. "How many dreams did Lincoln have in 1863?") |
| `EVAL_HARD_FACTUAL` | 15 | Obscure-but-real, verifiable facts (e.g. "Who was the first woman to win a Nobel Prize in Economics?") |
| `EVAL_HARD_UNCERTAIN` | 15 | Common misconceptions phrased as yes/no questions (e.g. "Did Einstein fail math in school?") |

Results: `results/hallucination/eval_results.json` (easy: precision=1.0, recall=1.0,
F1=1.0, avg_factual_score=-9.84, avg_halluc_score=+11.01) and
`results/hallucination/eval_hard_results.json` (hard: precision=0.0, recall=0.0, F1=0.0,
14 true negatives, 15 false negatives, 1 false positive out of 15+15).

## Dashboard tabs (`dashboard.py`)

7 tabs, confirmed directly from `st.tabs([...])`: Geometry, Steering Sweeps, Defense,
Scenario Validation, Hallucination, Cross-Model, Sample Responses. (The Scenario
Validation tab additionally splits into two sub-tabs: Self-Directed Scenarios and
External Scenarios.)

## Test suite

49 tests, 5 files, all pass without GPU (mocked layers): `test_model_adapter.py`,
`test_scenario_elicitation.py`, `test_steering.py`, `test_story_generator.py`,
`test_vector_extraction.py`. Run with `pytest tests/`.
