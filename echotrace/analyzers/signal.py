"""
Task A — Signal Analysis.

Computes SNR (Signal-to-Noise Ratio) and performs VAD
(Voice Activity Detection) using librosa energy analysis.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

import librosa
import numpy as np
from loguru import logger

from echotrace.core.telemetry import TelemetryCollector, timed


@dataclass
class VADSegment:
    start_ms: float
    end_ms: float


@dataclass
class SignalResult:
    snr_db: float
    vad_segments: List[VADSegment]
    silence_ratio: float
    duration_sec: float
    sample_rate: int
    is_too_short: bool = False


class SignalAnalyzer:
    """
    Analyzes audio signal quality and detects voice activity.

    Uses librosa RMS energy for SNR estimation and
    librosa.effects.split() for VAD boundary detection.
    """

    def __init__(self, collector: TelemetryCollector) -> None:
        self._collector = collector

    async def analyze(self, audio_path: str) -> SignalResult:
        """Run signal analysis in a thread pool (librosa is blocking)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._analyze_sync, audio_path)

    @timed("signal_analysis")
    async def analyze_timed(self, audio_path: str) -> SignalResult:
        """Timed version — call this from the orchestrator."""
        return await self.analyze(audio_path, _collector=self._collector)

    def _analyze_sync(self, audio_path: str) -> SignalResult:
        logger.info(f"Signal analysis: loading {audio_path}")

        y, sr = librosa.load(audio_path, sr=None, mono=True)
        duration_sec = librosa.get_duration(y=y, sr=sr)

        # Flag very short clips
        if duration_sec < 1.0:
            logger.warning("Clip < 1s — unreliable analysis")
            return SignalResult(
                snr_db=0.0,
                vad_segments=[],
                silence_ratio=1.0,
                duration_sec=duration_sec,
                sample_rate=sr,
                is_too_short=True,
            )

        # --- VAD using librosa.effects.split ---
        # top_db=30 is a reasonable default for speech detection
        intervals = librosa.effects.split(y, top_db=30)
        vad_segments = [
            VADSegment(
                start_ms=round(start / sr * 1000, 2),
                end_ms=round(end / sr * 1000, 2),
            )
            for start, end in intervals
        ]

        # --- Silence ratio ---
        speech_samples = sum(end - start for start, end in intervals)
        silence_ratio = 1.0 - (speech_samples / len(y)) if len(y) > 0 else 1.0

        # --- SNR Calculation ---
        snr_db = self._calculate_snr(y, intervals)

        logger.info(
            f"Signal result: SNR={snr_db:.1f}dB, "
            f"segments={len(vad_segments)}, "
            f"silence_ratio={silence_ratio:.2f}"
        )

        return SignalResult(
            snr_db=snr_db,
            vad_segments=vad_segments,
            silence_ratio=round(silence_ratio, 3),
            duration_sec=round(duration_sec, 3),
            sample_rate=sr,
        )

    def _calculate_snr(
        self, y: np.ndarray, speech_intervals: np.ndarray
    ) -> float:
        """
        Estimates SNR by comparing RMS energy of speech segments
        vs. non-speech (noise) segments.
        """
        if len(speech_intervals) == 0:
            return 0.0

        # Build masks
        speech_mask = np.zeros(len(y), dtype=bool)
        for start, end in speech_intervals:
            speech_mask[start:end] = True

        speech_signal = y[speech_mask]
        noise_signal = y[~speech_mask]

        if len(speech_signal) == 0:
            return 0.0
        if len(noise_signal) == 0:
            # All audio is speech — effectively infinite SNR
            return 100.0

        rms_speech = np.sqrt(np.mean(speech_signal**2))
        rms_noise = np.sqrt(np.mean(noise_signal**2))

        if rms_noise < 1e-10:
            rms_noise = 1e-10

        snr = 20 * np.log10(rms_speech / rms_noise)
        return round(float(snr), 2)
