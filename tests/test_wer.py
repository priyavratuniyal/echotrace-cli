"""Tests for WER computation and word-level diff."""

import pytest

from echotrace.analyzers.transcription import TranscriptionAuditor


class TestWER:
    """Test the Word Error Rate calculation."""

    def test_identical_strings(self):
        wer = TranscriptionAuditor._compute_wer(
            "hello world", "hello world"
        )
        assert wer == 0.0

    def test_completely_different(self):
        wer = TranscriptionAuditor._compute_wer(
            "hello world", "foo bar"
        )
        assert wer == 1.0

    def test_one_substitution(self):
        wer = TranscriptionAuditor._compute_wer(
            "I want to book a comprehensive heart checkup",
            "I want to book a compassion heart checkup",
        )
        # 1 substitution out of 9 words ≈ 0.111
        assert 0.1 < wer < 0.15

    def test_insertion(self):
        wer = TranscriptionAuditor._compute_wer(
            "hello world",
            "hello beautiful world",
        )
        # 1 insertion / 2 reference words = 0.5
        assert wer == pytest.approx(0.5, abs=0.01)

    def test_deletion(self):
        wer = TranscriptionAuditor._compute_wer(
            "hello beautiful world",
            "hello world",
        )
        # 1 deletion / 3 reference words ≈ 0.333
        assert wer == pytest.approx(1 / 3, abs=0.01)

    def test_empty_reference(self):
        wer = TranscriptionAuditor._compute_wer("", "hello")
        assert wer == 1.0

    def test_empty_hypothesis(self):
        wer = TranscriptionAuditor._compute_wer("hello world", "")
        assert wer == 1.0

    def test_both_empty(self):
        wer = TranscriptionAuditor._compute_wer("", "")
        assert wer == 0.0

    def test_case_insensitive(self):
        wer = TranscriptionAuditor._compute_wer(
            "Hello World", "hello world"
        )
        assert wer == 0.0


class TestDiff:
    """Test the word-level diff generation."""

    def test_perfect_match(self):
        diff = TranscriptionAuditor._compute_diff(
            "hello world", "hello world"
        )
        assert all(d.status == "correct" for d in diff)

    def test_substitution(self):
        diff = TranscriptionAuditor._compute_diff(
            "I want comprehensive checkup",
            "I want compassion checkup",
        )
        subs = [d for d in diff if d.status == "substituted"]
        assert len(subs) == 1
        assert subs[0].word == "comprehensive"
        assert subs[0].actual == "compassion"

    def test_deletion(self):
        diff = TranscriptionAuditor._compute_diff(
            "hello beautiful world",
            "hello world",
        )
        dels = [d for d in diff if d.status == "deleted"]
        assert len(dels) == 1
        assert dels[0].word == "beautiful"

    def test_insertion(self):
        diff = TranscriptionAuditor._compute_diff(
            "hello world",
            "hello beautiful world",
        )
        ins = [d for d in diff if d.status == "inserted"]
        assert len(ins) == 1
        assert ins[0].word == "beautiful"
