"""Tests for vlm_defect.evaluate helper functions.

Focuses on the pure, side-effect-free helpers that are safe to run without
GPU, model weights, or external dependencies.
"""

from __future__ import annotations

import pytest

from vlm_defect.evaluate import (
    _compute_metrics,
    _extract_defect_name,
    _fuzzy_score,
    _is_anomaly_response,
)


# ---------------------------------------------------------------------------
# _is_anomaly_response
# ---------------------------------------------------------------------------

class TestIsAnomalyResponse:
    # ── clear positives ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "Yes.",
        "Yes",
        "YES",
        "yes",
        "Yes, there is a broken_large anomaly.",
        "Yes, there is an contamination anomaly.",
        "Yes, but I'm not entirely sure.",   # hedged yes still counts
        "  yes  ",                           # leading/trailing whitespace
        "YES — multiple defects detected.",
    ])
    def test_true_for_yes_variants(self, text):
        assert _is_anomaly_response(text) is True

    # ── clear negatives ──────────────────────────────────────────────────────
    @pytest.mark.parametrize("text", [
        "No.",
        "No",
        "NO",
        "no",
        "No anomaly detected.",
        "No, the surface looks intact.",
        "No, wait — actually there is something.",  # starts with No
        "",
        "   ",
    ])
    def test_false_for_no_variants(self, text):
        assert _is_anomaly_response(text) is False

    # ── word-boundary regression: "yes" must not match mid-word ─────────────
    @pytest.mark.parametrize("text", [
        "Yesterday the inspection was clear.",
        "YESTERDAY",
        "Yessir, no defects here.",   # uncommon but shouldn't match as "Yes"
    ])
    def test_false_for_yes_as_word_prefix(self, text):
        """'YESTERDAY' starts with 'yes' but is not a Yes response."""
        assert _is_anomaly_response(text) is False

    def test_empty_string_is_false(self):
        assert _is_anomaly_response("") is False

    def test_whitespace_only_is_false(self):
        assert _is_anomaly_response("   \t\n") is False


# ---------------------------------------------------------------------------
# _extract_defect_name
# ---------------------------------------------------------------------------

class TestExtractDefectName:
    @pytest.mark.parametrize("text, expected", [
        (
            "Yes, there is a broken_large anomaly.",
            "broken_large",
        ),
        (
            "Yes, there is an contamination anomaly.",
            "contamination",
        ),
        (
            "Yes, there is a scratch anomaly present.",
            "scratch",
        ),
        (
            # Case-insensitive pattern match
            "Yes, THERE IS A BENT anomaly.",
            "BENT",
        ),
        (
            # Embedded in a longer sentence
            "The image shows defects. There is a hole anomaly clearly visible.",
            "hole",
        ),
    ])
    def test_extracts_known_patterns(self, text, expected):
        assert _extract_defect_name(text) == expected

    @pytest.mark.parametrize("text", [
        "No.",
        "No anomaly detected.",
        "Yes, there is some damage.",        # "damage" not followed by "anomaly"
        "The surface appears normal.",
        "",
    ])
    def test_returns_none_when_no_match(self, text):
        assert _extract_defect_name(text) is None

    def test_strips_whitespace_from_result(self):
        # Ensure internal spaces in the capture group are stripped
        result = _extract_defect_name("Yes, there is a  crack  anomaly.")
        assert result is not None
        assert result == result.strip()


# ---------------------------------------------------------------------------
# _fuzzy_score
# ---------------------------------------------------------------------------

class TestFuzzyScore:
    def test_identical_strings_score_one(self):
        assert _fuzzy_score("broken_large", "broken_large") == pytest.approx(1.0)

    def test_completely_different_strings_score_less_than_half(self):
        score = _fuzzy_score("scratch", "contamination")
        assert score < 0.5

    def test_similar_strings_score_between_zero_and_one(self):
        score = _fuzzy_score("broken_large", "broken_small")
        assert 0.0 < score < 1.0

    def test_score_is_symmetric(self):
        a, b = "bent", "bent_inward"
        assert _fuzzy_score(a, b) == pytest.approx(_fuzzy_score(b, a))

    def test_returns_float(self):
        assert isinstance(_fuzzy_score("a", "b"), float)


# ---------------------------------------------------------------------------
# _compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_classifier(self):
        m = _compute_metrics(tp=10, fp=0, tn=10, fn=0)
        assert m["accuracy"]  == pytest.approx(1.0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["f1"]        == pytest.approx(1.0)
        assert m["n_samples"] == 20

    def test_all_false_positives(self):
        # Predicts everything as anomaly, nothing is actually anomaly
        m = _compute_metrics(tp=0, fp=10, tn=0, fn=0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"]    == pytest.approx(0.0)  # no true positives
        assert m["accuracy"]  == pytest.approx(0.0)

    def test_all_false_negatives(self):
        # Predicts everything as normal, everything is actually anomaly
        m = _compute_metrics(tp=0, fp=0, tn=0, fn=10)
        assert m["recall"]    == pytest.approx(0.0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["f1"]        == pytest.approx(0.0)

    def test_zero_division_no_predicted_positives(self):
        """Precision is 0.0 when the model never predicts anomaly."""
        m = _compute_metrics(tp=0, fp=0, tn=5, fn=5)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"]    == pytest.approx(0.0)
        assert m["f1"]        == pytest.approx(0.0)

    def test_zero_division_no_samples_at_all(self):
        m = _compute_metrics(tp=0, fp=0, tn=0, fn=0)
        assert m["accuracy"]  == pytest.approx(0.0)
        assert m["n_samples"] == 0

    def test_confusion_matrix_values_preserved(self):
        m = _compute_metrics(tp=3, fp=1, tn=4, fn=2)
        assert m["confusion_matrix"] == {"tp": 3, "fp": 1, "tn": 4, "fn": 2}

    def test_n_samples_is_sum_of_quadruple(self):
        m = _compute_metrics(tp=3, fp=1, tn=4, fn=2)
        assert m["n_samples"] == 10

    def test_f1_harmonic_mean_of_precision_and_recall(self):
        # precision = 2/(2+1) = 2/3, recall = 2/(2+2) = 1/2
        # f1 = 2 * (2/3 * 1/2) / (2/3 + 1/2) = 2 * (1/3) / (7/6) = (2/3)/(7/6) = 4/7
        m = _compute_metrics(tp=2, fp=1, tn=5, fn=2)
        expected_f1 = 4 / 7
        assert m["f1"] == pytest.approx(expected_f1, abs=1e-3)

    def test_output_values_are_rounded_to_four_decimal_places(self):
        m = _compute_metrics(tp=1, fp=2, tn=3, fn=4)
        for key in ("accuracy", "precision", "recall", "f1"):
            val = m[key]
            assert val == round(val, 4)
