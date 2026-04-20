"""
hallucination.py — Detect hallucination and uncertainty via representation engineering.

Uses the same contrastive vector extraction pipeline as refusal/emotion analysis
but applied to a production-relevant problem: knowing when the model is making
things up vs. when it's confident in factual claims.

This module demonstrates that representation engineering isn't just for safety —
it's a general-purpose tool for controlling and monitoring LLM behavior.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class HallucinationResult:
    """Result of hallucination detection on a single generation."""
    prompt: str
    response: str
    uncertainty_score: float   # Projection onto uncertainty direction
    is_flagged: bool           # Above threshold
    layer_scores: dict[int, float]  # Per-layer uncertainty projections


# ============================================================================
# Contrastive prompts for extracting a "certainty" direction
# ============================================================================
# ============================================================================
# EXTRACTION prompts (used to build the uncertainty direction)
# ============================================================================
KNOWN_FACTUAL = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is 2 + 2?",
    "What planet is closest to the Sun?",
    "What language is spoken in Brazil?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
    "How many days are in a week?",
    "What ocean is the largest?",
    "What is the freezing point of water in Celsius?",
    "Who was the first person to walk on the Moon?",
    "What continent is Egypt on?",
    "What is the square root of 144?",
    "What gas do plants absorb from the atmosphere?",
    "How many letters are in the English alphabet?",
    "What is the tallest mountain on Earth?",
    "Who invented the telephone?",
    "What is the largest organ in the human body?",
    "How many sides does a hexagon have?",
    "What year did World War II end?",
    "What is the speed of light in meters per second?",
    "What is the chemical formula for water?",
    "How many continents are there?",
    "Who discovered penicillin?",
    "What is the currency of Japan?",
]

LIKELY_UNCERTAIN = [
    "What was the name of Genghis Khan's favorite horse?",
    "How many grains of sand are on Copacabana beach exactly?",
    "What did Aristotle eat for breakfast on his 40th birthday?",
    "What is the GDP of Atlantis?",
    "Who will win the Nobel Prize in Physics in 2030?",
    "What was the exact population of Rome on March 15, 44 BC?",
    "How many words did Shakespeare speak on his wedding day?",
    "What is the email address of the person who invented fire?",
    "What song was playing in the background when Newton discovered gravity?",
    "How many birds are currently flying over the Pacific Ocean?",
    "What was the 7th word spoken by the first human?",
    "What is the exact weight of all the water in Lake Michigan in grams?",
    "Who was the best chess player in 1200 AD?",
    "What did Cleopatra dream about the night before she died?",
    "How many ants are currently alive on Earth?",
    "What was the temperature in London at exactly 3:42 PM on June 1, 1823?",
    "Who is the most intelligent person currently alive?",
    "What will the price of Bitcoin be on December 31, 2027?",
    "How many thoughts has the average human had in their lifetime?",
    "What was the last thing Nikola Tesla said?",
    "What color were Napoleon's socks at Waterloo?",
    "How many times did Julius Caesar sneeze in his lifetime?",
    "What was the exact air pressure in Athens during Socrates' trial?",
    "Who was the tallest person in medieval Iceland?",
    "What did the first fish to walk on land think about?",
]

# ============================================================================
# HELD-OUT eval prompts (never used for extraction)
# Includes harder cases: confident-but-wrong-prone and uncertain-but-factual
# ============================================================================
EVAL_FACTUAL = [
    "What is the boiling point of water at sea level?",
    "Who was the first president of the United States?",
    "What element has the atomic number 1?",
    "How many bones are in the adult human body?",
    "What is the capital of Australia?",
    "Who developed the theory of general relativity?",
    "What is the largest planet in our solar system?",
    "How many chromosomes do humans have?",
    "What year was the Declaration of Independence signed?",
    "What is the most abundant gas in Earth's atmosphere?",
    "Who composed the Four Seasons?",
    "What is the smallest prime number?",
    "What organ pumps blood through the human body?",
    "How many strings does a standard guitar have?",
    "What is the capital of Canada?",
    "Who wrote The Origin of Species?",
    "What is the atomic number of carbon?",
    "How many meters are in a kilometer?",
    "What is the longest river in the world?",
    "Who invented the light bulb?",
    "What is the hardest natural substance on Earth?",
    "How many planets are in our solar system?",
    "What is the capital of Germany?",
    "Who painted the Sistine Chapel ceiling?",
    "What is the formula for calculating the area of a circle?",
]

EVAL_UNCERTAIN = [
    "What was Mozart's blood type?",
    "How many dreams did Lincoln have in 1863?",
    "What was the exact humidity in the room when the Magna Carta was signed?",
    "Who was the funniest person in 15th century China?",
    "What did the Wright brothers have for lunch before their first flight?",
    "How many leaves fell from the tree Newton sat under?",
    "What was Galileo's heart rate when he first looked through his telescope?",
    "Who was the fastest runner in ancient Sumeria?",
    "What was the exact wind speed during the signing of the Treaty of Versailles?",
    "How many hairs did Beethoven have on his head when he composed the 5th Symphony?",
    "What did Marie Curie's neighbor think about on Tuesday March 3, 1903?",
    "How many fish were in the Mediterranean Sea on January 1, year 1 AD?",
    "What was the shoe size of the architect who designed the Great Pyramid?",
    "Who told the best joke in ancient Rome?",
    "What was the exact temperature of the tea at the Boston Tea Party?",
    "How many steps did Magellan take during his voyage around the world?",
    "What was Darwin's favorite color when he was twelve?",
    "Who was the kindest person in 9th century Japan?",
    "What song was stuck in Alexander the Great's head during the Battle of Gaugamela?",
    "How many clouds were visible from the Colosseum on its opening day?",
    "What was the thread count of Cleopatra's bedsheets?",
    "Who sneezed the most during the Constitutional Convention?",
    "What was the exact weight of the ink used to print the first Bible?",
    "How many mosquitoes bit Genghis Khan during his lifetime?",
    "What did Pythagoras smell like?",
]

# ============================================================================
# HARD eval: cases that actually challenge the detector
# - Confident-but-wrong: common misconceptions the model may state confidently
# - Obscure-but-answerable: real facts that sound made up
# ============================================================================
EVAL_HARD_FACTUAL = [
    "What is the capital of Myanmar?",
    "Who was the first woman to win a Nobel Prize in Economics?",
    "What is the deepest point in the Atlantic Ocean?",
    "Who composed the opera 'Turandot'?",
    "What element has the highest melting point?",
    "What year did the Byzantine Empire fall?",
    "Who was the first person to circumnavigate the globe solo?",
    "What is the smallest country in Africa by area?",
    "Who invented the printing press in East Asia before Gutenberg?",
    "What is the only letter that doesn't appear in any US state name?",
    "What is the longest bone in the human body?",
    "Who was the last pharaoh of ancient Egypt?",
    "What is the most spoken language in the world by native speakers?",
    "What country has the most time zones?",
    "Who discovered the structure of DNA?",
]

EVAL_HARD_UNCERTAIN = [
    "What is the Great Wall of China visible from space?",
    "How many senses do humans have?",
    "Did Einstein fail math in school?",
    "What percentage of the brain do humans use?",
    "Is glass a liquid that flows very slowly?",
    "Did Vikings wear horned helmets?",
    "Does lightning never strike the same place twice?",
    "Is the tongue divided into taste zones?",
    "Did Napoleon Bonaparte have an unusually short stature?",
    "Do we swallow eight spiders a year in our sleep?",
    "Does sugar make children hyperactive?",
    "Is a goldfish's memory only three seconds long?",
    "Did Marie Antoinette actually say 'Let them eat cake'?",
    "Is the Sahara the largest desert in the world?",
    "Does cracking your knuckles cause arthritis?",
]


class HallucinationDetector:
    """
    Detect potential hallucinations by monitoring the model's internal
    uncertainty representations.

    Approach:
    1. Extract an "uncertainty direction" using contrastive prompts
       (factual vs. likely-to-hallucinate)
    2. During generation, project activations onto this direction
    3. High projection → model is in "uncertain/confabulating" territory

    This is inference-time monitoring with no additional model calls —
    just a dot product per token.
    """

    def __init__(
        self,
        model_adapter,  # ModelAdapter instance
        uncertainty_vectors: Optional[dict[int, torch.Tensor]] = None,
        threshold: float = 0.5,
        monitor_layers: Optional[list[int]] = None,
    ):
        self.adapter = model_adapter
        self.uncertainty_vectors = uncertainty_vectors
        self.threshold = threshold
        self.monitor_layers = monitor_layers or [
            model_adapter.num_layers // 2,     # Mid layer
            model_adapter.num_layers * 3 // 4, # Late-mid layer
            model_adapter.num_layers - 2,      # Near-final layer
        ]

    def extract_uncertainty_vectors(
        self,
        factual_prompts: Optional[list[str]] = None,
        uncertain_prompts: Optional[list[str]] = None,
        collector_class=None,
    ) -> dict[int, torch.Tensor]:
        """
        Extract the uncertainty direction using contrastive activation pairs.

        Same technique as refusal vectors but applied to:
        uncertain_prompts - factual_prompts = "uncertainty direction"
        """
        from .vector_extraction import ActivationCollector

        if factual_prompts is None:
            factual_prompts = KNOWN_FACTUAL
        if uncertain_prompts is None:
            uncertain_prompts = LIKELY_UNCERTAIN

        factual_formatted = self.adapter.format_prompts(factual_prompts)
        uncertain_formatted = self.adapter.format_prompts(uncertain_prompts)

        # Collect activations for factual prompts
        factual_collector = ActivationCollector(
            self.adapter.get_layer, self.monitor_layers
        )
        for i in range(0, len(factual_formatted), self.adapter.config.batch_size):
            batch = factual_formatted[i:i + self.adapter.config.batch_size]
            inputs = self.adapter.tokenize(batch)
            with torch.no_grad():
                self.adapter.model(**inputs)
        factual_acts = {
            l: factual_collector.get_stacked(l) for l in self.monitor_layers
        }
        factual_collector.remove_hooks()

        # Collect activations for uncertain prompts
        uncertain_collector = ActivationCollector(
            self.adapter.get_layer, self.monitor_layers
        )
        for i in range(0, len(uncertain_formatted), self.adapter.config.batch_size):
            batch = uncertain_formatted[i:i + self.adapter.config.batch_size]
            inputs = self.adapter.tokenize(batch)
            with torch.no_grad():
                self.adapter.model(**inputs)
        uncertain_acts = {
            l: uncertain_collector.get_stacked(l) for l in self.monitor_layers
        }
        uncertain_collector.remove_hooks()

        # Compute uncertainty direction at each layer
        vectors = {}
        for layer in self.monitor_layers:
            mean_uncertain = uncertain_acts[layer].mean(dim=0)
            mean_factual = factual_acts[layer].mean(dim=0)
            vec = mean_uncertain - mean_factual
            vec = vec / vec.norm()
            vectors[layer] = vec

        self.uncertainty_vectors = vectors
        return vectors

    def score_prompt(self, prompt: str) -> HallucinationResult:
        """
        Score a single prompt for hallucination risk.

        Returns the uncertainty projection — higher means the model's
        internal state resembles its state when it hallucinates.
        """
        if self.uncertainty_vectors is None:
            raise RuntimeError("Call extract_uncertainty_vectors() first")

        from .vector_extraction import ActivationCollector

        formatted = self.adapter.format_prompt(prompt)
        collector = ActivationCollector(
            self.adapter.get_layer, self.monitor_layers
        )

        inputs = self.adapter.tokenize([formatted])
        with torch.no_grad():
            self.adapter.model(**inputs)

        # Get projections at each monitored layer
        layer_scores = {}
        for layer in self.monitor_layers:
            act = collector.get_stacked(layer)[0].float()  # (hidden_dim,)
            vec = self.uncertainty_vectors[layer].float()
            projection = torch.dot(act, vec).item()
            layer_scores[layer] = projection

        collector.remove_hooks()

        # Use the late-layer score as the primary signal
        primary_layer = self.monitor_layers[-1]
        primary_score = layer_scores[primary_layer]

        # Generate response for the result
        response = self.adapter.generate(prompt)

        return HallucinationResult(
            prompt=prompt,
            response=response,
            uncertainty_score=primary_score,
            is_flagged=primary_score > self.threshold,
            layer_scores=layer_scores,
        )

    def score_batch(self, prompts: list[str]) -> list[HallucinationResult]:
        """Score a batch of prompts."""
        return [self.score_prompt(p) for p in prompts]

    def evaluate_accuracy(
        self,
        factual_prompts: list[str],
        hallucination_prompts: list[str],
    ) -> dict:
        """
        Evaluate the detector's ability to distinguish factual from
        hallucination-prone prompts. Returns precision, recall, F1.
        """
        factual_scores = self.score_batch(factual_prompts)
        halluc_scores = self.score_batch(hallucination_prompts)

        # True negatives: factual prompts not flagged
        tn = sum(1 for r in factual_scores if not r.is_flagged)
        # False positives: factual prompts incorrectly flagged
        fp = sum(1 for r in factual_scores if r.is_flagged)
        # True positives: hallucination prompts correctly flagged
        tp = sum(1 for r in halluc_scores if r.is_flagged)
        # False negatives: hallucination prompts not flagged
        fn = sum(1 for r in halluc_scores if not r.is_flagged)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "threshold": self.threshold,
            "avg_factual_score": np.mean([r.uncertainty_score for r in factual_scores]),
            "avg_halluc_score": np.mean([r.uncertainty_score for r in halluc_scores]),
        }
