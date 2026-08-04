# Future Directions

Concrete, unresolved items — each tied to something specific this project's own results
or code left open, not generic "future work" filler.

## 1. Rerun the headline sweeps against the tightened refusal classifier

All numbers in README sections 2, 3, and 6 (guilt -10%, confidence -6%, defense
94%→98%, Qwen 92-100%) were measured against an earlier, broader refusal-indicator
wordlist in `measure_refusal_rate` (`src/steering.py`). An independent review pass later
removed bare-word indicators ("harmful", "dangerous", "illegal", "instead, I") that
false-positive on compliant answers merely discussing or hedging around the topic,
keeping only explicit refusal phrasing. That fix hasn't been re-run on GPU yet — the
direction of the findings (guilt/confidence erode refusal more than desperation; Qwen is
steering-resistant) is very unlikely to flip, since the effect sizes were already
measured as real percentage-point swings rather than borderline classification calls,
but the exact percentages should be treated as provisional until re-measured.

## 2. Rerun emotion vector extraction against the padding-mask fix

The same review pass found that `mean_from_n` token-pooling (used for story-based
emotion vector extraction) averaged over left-padded batches without using the
attention mask, so pad/EOS token activations could contaminate the mean for any story
shorter than the batch's longest sequence. This was fixed with attention-mask-aware
pooling in `ActivationCollector` (`src/vector_extraction.py`). The saved
`results/emotion_refusal/emotion_vectors.pt` predates this fix — extraction hasn't been
rerun. Expected impact: magnitude noise reduction, not a sign flip (padding contamination
adds a small, roughly emotion-independent bias toward the pad token's activation, which
shouldn't systematically favor one emotion's direction over another) — but this is a
prediction, not a re-verified result.

## 3. Why don't fear and desperation activate under self-directed pressure?

Self-directed validation scenarios produced 4/6 diagonal matches (calm, anger, guilt,
confidence) but fear and desperation still didn't activate their own target vector most
strongly (`results/emotion_refusal/self_directed_validation_avg.json`). Two untested
hypotheses: (a) the story-based fear/desperation vectors may be capturing a narrower or
more context-specific slice of the concept than the scenario prompts elicit, or (b)
these two emotions may need scenarios with even more direct self-referential framing
than what `SELF_DIRECTED_SCENARIOS` currently provides. Neither has been tested.

## 4. Why is Qwen's safety training steering-resistant when the geometry is the same?

Section 6 shows 6/6 sign agreement on cosine similarity (same emotion-refusal geometry
in both models) but Qwen's refusal rate stays in a 92-100% band regardless of steering
strength, while Llama shows real erosion at the same relative strengths. This project
doesn't investigate *why* — candidate explanations (different safety fine-tuning data,
different residual-stream norm scaling meaning the same additive strength represents a
smaller relative perturbation in Qwen, a wider "refusal cone" per the Concept Cones
framing referenced in the README) are all untested. Comparing residual-stream norms
per-layer between the two models before drawing conclusions would be the first concrete
step.

## 5. `scenario_elicitation.py`'s `token_mode` is hardcoded to `"last"`

Noted directly in the project's own working notes as low-priority but unexplored:
comparing last-token vs. mean-pooled projections during scenario validation might change
which scenarios activate which vectors, particularly for the fear/desperation gap in
item 3 above.

## 6. Outstanding from the original task list

Per the project's own tracked task list: a LinkedIn writeup summarizing the findings for
a general audience, and a final pass confirming the public GitHub repo is clean (no
stray credentials/large binaries, `.gitignore` correctly scoped) — both still open as of
this writing.
