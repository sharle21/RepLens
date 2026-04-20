"""Tests for scenario_elicitation.py — validation logic without GPU."""

import pytest
from src.scenario_elicitation import (
    VALIDATION_SCENARIOS,
    ValidationResult,
    summarize_validation,
    print_validation_matrix,
)


class TestValidationScenarios:
    def test_all_emotions_present(self):
        expected = {"desperation", "calm", "anger", "fear", "guilt", "confidence"}
        assert set(VALIDATION_SCENARIOS.keys()) == expected

    def test_each_emotion_has_10_scenarios(self):
        for emotion, scenarios in VALIDATION_SCENARIOS.items():
            assert len(scenarios) == 10, f"{emotion} has {len(scenarios)} scenarios, expected 10"

    def test_scenarios_are_nonempty_strings(self):
        for emotion, scenarios in VALIDATION_SCENARIOS.items():
            for i, s in enumerate(scenarios):
                assert isinstance(s, str) and len(s) > 20, (
                    f"{emotion}[{i}] is too short or not a string"
                )


class TestSummarizeValidation:
    def _make_results(self):
        """Create fake validation results with known projections."""
        results = []
        # desperation scenarios score high on desperation, low on calm
        for i in range(3):
            results.append(ValidationResult(
                scenario_text=f"desp scenario {i}",
                scenario_emotion="desperation",
                projections={"desperation": 2.0, "calm": 0.5},
                layer=16,
            ))
        # calm scenarios score high on calm, low on desperation
        for i in range(3):
            results.append(ValidationResult(
                scenario_text=f"calm scenario {i}",
                scenario_emotion="calm",
                projections={"desperation": 0.3, "calm": 1.8},
                layer=16,
            ))
        return results

    def test_summary_has_correct_structure(self):
        results = self._make_results()
        summary = summarize_validation(results)

        assert "desperation" in summary
        assert "calm" in summary
        assert "desperation" in summary["desperation"]
        assert "calm" in summary["desperation"]

    def test_summary_values_are_means(self):
        results = self._make_results()
        summary = summarize_validation(results)

        assert abs(summary["desperation"]["desperation"] - 2.0) < 1e-6
        assert abs(summary["desperation"]["calm"] - 0.5) < 1e-6
        assert abs(summary["calm"]["calm"] - 1.8) < 1e-6

    def test_layer_filter(self):
        results = self._make_results()
        # Add results at a different layer
        results.append(ValidationResult(
            scenario_text="desp at layer 20",
            scenario_emotion="desperation",
            projections={"desperation": 99.0, "calm": 99.0},
            layer=20,
        ))

        # Filter to layer 16 only — should exclude layer 20
        summary = summarize_validation(results, layer=16)
        assert abs(summary["desperation"]["desperation"] - 2.0) < 1e-6

    def test_empty_results(self):
        summary = summarize_validation([])
        assert summary == {}


class TestPrintValidationMatrix:
    def test_prints_without_error(self, capsys):
        summary = {
            "desperation": {"desperation": 2.0, "calm": 0.5},
            "calm": {"desperation": 0.3, "calm": 1.8},
        }
        print_validation_matrix(summary)
        captured = capsys.readouterr()
        assert "desperation" in captured.out
        assert "calm" in captured.out

    def test_empty_summary(self, capsys):
        print_validation_matrix({})
        captured = capsys.readouterr()
        assert "No results" in captured.out
