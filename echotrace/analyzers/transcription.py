"""
Task B — Transcription Audit.

Runs faster-whisper in two modes (tiny + large-v2) and computes
Word Error Rate (WER) between the "actual" (tiny) and "gold standard"
(large-v2 or user-provided reference).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from echotrace.core.telemetry import TelemetryCollector, timed


@dataclass
class DiffToken:
    """A single word-level diff entry."""

    word: str
    status: str  # "correct" | "substituted" | "deleted" | "inserted"
    actual: Optional[str] = None  # only for substitutions


@dataclass
class TranscriptionResult:
    tiny_transcript: str
    gold_transcript: str
    wer: Optional[float]
    stt_latency_ms: float  # wall-clock time for the tiny model
    diff: List[DiffToken]
    gold_source: str  # "whisper-large-v2" | "user-reference"
    no_speech: bool = False


class TranscriptionAuditor:
    """
    Dual-model transcription auditor.

    Runs Whisper tiny (simulating production STT) and optionally
    Whisper large-v2 (gold standard). If a user reference is provided,
    it is used as gold standard instead of large-v2.
    """

    DEFAULT_GOLD_MODEL = "large-v2"
    SUGGESTED_MODELS = [
        ("large-v2",            "~3 GB",  "Best accuracy, slow"),
        ("medium",              "~1.5 GB", "Good balance of speed and accuracy"),
        ("small",               "~500 MB", "Fast, decent accuracy"),
        ("distil-large-v3",     "~1.5 GB", "Distilled, faster than large-v2"),
    ]

    def __init__(
        self,
        collector: TelemetryCollector,
        reference_text: Optional[str] = None,
        gold_model_name: Optional[str] = None,
    ) -> None:
        self._collector = collector
        self._reference_text = reference_text
        self._tiny_model = None
        self._gold_model = None
        self._gold_model_name: str = gold_model_name or self.DEFAULT_GOLD_MODEL

    def _load_tiny(self):
        if self._tiny_model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading primary STT model (tiny)...")
            self._tiny_model = WhisperModel(
                "tiny", device="cpu", compute_type="int8"
            )
        return self._tiny_model

    def _load_gold(self) -> object:
        """Load the gold-standard Whisper model."""
        if self._gold_model is None:
            from faster_whisper import WhisperModel
            logger.info(f"Loading reference model ({self._gold_model_name})...")
            self._gold_model = WhisperModel(
                self._gold_model_name, device="cpu", compute_type="int8"
            )
        return self._gold_model

    async def audit(self, audio_path: str) -> TranscriptionResult:
        """Run the full transcription audit pipeline."""
        import time as _time

        loop = asyncio.get_event_loop()

        # Always run tiny — record timing separately
        t0 = _time.perf_counter()
        tiny_result = await loop.run_in_executor(
            None, self._transcribe_tiny, audio_path
        )
        t1 = _time.perf_counter()
        await self._collector.record("stt_tiny", t0, t1)

        # Determine gold standard
        if self._reference_text:
            gold_text = self._reference_text
            gold_source = "user-reference"
        else:
            t2 = _time.perf_counter()
            gold_text = await loop.run_in_executor(
                None, self._transcribe_gold, audio_path
            )
            t3 = _time.perf_counter()
            await self._collector.record("stt_gold", t2, t3)
            if "/" not in self._gold_model_name:
                gold_source = f"Systran/faster-whisper-{self._gold_model_name}"
            else:
                gold_source = self._gold_model_name

        tiny_text = tiny_result["text"]
        stt_latency_ms = tiny_result["latency_ms"]

        # Handle no-speech edge case
        if not tiny_text.strip() and not gold_text.strip():
            return TranscriptionResult(
                tiny_transcript="",
                gold_transcript="",
                wer=None,
                stt_latency_ms=stt_latency_ms,
                diff=[],
                gold_source=gold_source,
                no_speech=True,
            )

        # Compute WER
        wer = self._compute_wer(gold_text, tiny_text)

        # Compute word-level diff
        diff = self._compute_diff(gold_text, tiny_text)

        logger.info(f"Transcription audit: WER={wer:.3f}, source={gold_source}")

        return TranscriptionResult(
            tiny_transcript=tiny_text,
            gold_transcript=gold_text,
            wer=round(wer, 4),
            stt_latency_ms=stt_latency_ms,
            diff=diff,
            gold_source=gold_source,
        )

    @timed("stt_tiny")
    async def _run_tiny_timed(self, audio_path: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._transcribe_tiny, audio_path, _collector=self._collector
        )

    def _transcribe_tiny(self, audio_path: str) -> dict:
        import time

        model = self._load_tiny()
        start = time.perf_counter()
        segments, info = model.transcribe(audio_path, beam_size=5)
        text = " ".join(s.text for s in segments).strip()
        elapsed = (time.perf_counter() - start) * 1000

        return {"text": text, "latency_ms": round(elapsed, 2)}

    def _transcribe_gold(self, audio_path: str) -> str:
        model = self._load_gold()
        segments, _ = model.transcribe(audio_path, beam_size=5)
        return " ".join(s.text for s in segments).strip()

    @staticmethod
    def _normalize_text(text: str) -> str:
        import string
        words = text.lower().split()
        cleaned_words = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]
        return " ".join(cleaned_words)

    @staticmethod
    def _compute_wer(reference: str, hypothesis: str) -> float:
        """
        Compute Word Error Rate.
        Formula: (S + D + I) / N
        Uses jiwer for robust computation.
        """
        ref_clean = TranscriptionAuditor._normalize_text(reference)
        hyp_clean = TranscriptionAuditor._normalize_text(hypothesis)

        try:
            from jiwer import wer

            return wer(ref_clean, hyp_clean)
        except Exception:
            # Manual fallback
            ref_words = ref_clean.split()
            hyp_words = hyp_clean.split()
            if not ref_words:
                return 0.0 if not hyp_words else 1.0
            return TranscriptionAuditor._levenshtein_wer(ref_words, hyp_words)

    @staticmethod
    def _levenshtein_wer(ref: List[str], hyp: List[str]) -> float:
        """Manual WER via edit distance on word lists."""
        n = len(ref)
        m = len(hyp)
        dp = [[0] * (m + 1) for _ in range(n + 1)]

        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],      # deletion
                        dp[i][j - 1],      # insertion
                        dp[i - 1][j - 1],  # substitution
                    )

        return dp[n][m] / n if n > 0 else 0.0

    @staticmethod
    def _compute_diff(
        reference: str, hypothesis: str
    ) -> List[DiffToken]:
        """
        Produce a word-level diff between reference (gold)
        and hypothesis (actual).
        """
        ref_words = TranscriptionAuditor._normalize_text(reference).split()
        hyp_words = TranscriptionAuditor._normalize_text(hypothesis).split()
        diff: List[DiffToken] = []

        n, m = len(ref_words), len(hyp_words)
        # Build edit distance matrix + backtrack
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]
                    )

        # Backtrack to build diff
        i, j = n, m
        ops: list = []
        while i > 0 or j > 0:
            if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
                ops.append(("correct", ref_words[i - 1], None))
                i -= 1
                j -= 1
            elif (
                i > 0
                and j > 0
                and dp[i][j] == dp[i - 1][j - 1] + 1
            ):
                ops.append(("substituted", ref_words[i - 1], hyp_words[j - 1]))
                i -= 1
                j -= 1
            elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
                ops.append(("inserted", hyp_words[j - 1], None))
                j -= 1
            elif i > 0:
                ops.append(("deleted", ref_words[i - 1], None))
                i -= 1

        ops.reverse()
        for status, word, actual in ops:
            diff.append(DiffToken(word=word, status=status, actual=actual))

        return diff
