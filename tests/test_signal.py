"""Tests for signal analysis: SNR and VAD."""

import numpy as np
import pytest
import scipy.io.wavfile as wav
import tempfile
import os

from echotrace.analyzers.signal import SignalAnalyzer
from echotrace.core.telemetry import TelemetryCollector


@pytest.fixture
def collector():
    return TelemetryCollector()


@pytest.fixture
def tone_wav():
    """Generate a clean 440Hz tone WAV file."""
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_int16 = np.int16(audio * 32767)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sr, audio_int16)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def silent_wav():
    """Generate a completely silent WAV file."""
    sr = 16000
    duration = 2.0
    audio = np.zeros(int(sr * duration), dtype=np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sr, audio)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def short_wav():
    """Generate a very short (<1s) WAV file."""
    sr = 16000
    duration = 0.3
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio_int16 = np.int16(audio * 32767)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sr, audio_int16)
        yield f.name
    os.unlink(f.name)


class TestSignalAnalyzer:
    @pytest.mark.asyncio
    async def test_tone_has_positive_snr(self, collector, tone_wav):
        analyzer = SignalAnalyzer(collector)
        result = await analyzer.analyze(tone_wav)
        assert result.snr_db > 0

    @pytest.mark.asyncio
    async def test_short_clip_flagged(self, collector, short_wav):
        analyzer = SignalAnalyzer(collector)
        result = await analyzer.analyze(short_wav)
        assert result.is_too_short is True

    @pytest.mark.asyncio
    async def test_duration_reported(self, collector, tone_wav):
        analyzer = SignalAnalyzer(collector)
        result = await analyzer.analyze(tone_wav)
        assert 1.5 < result.duration_sec < 2.5

    @pytest.mark.asyncio
    async def test_sample_rate_reported(self, collector, tone_wav):
        analyzer = SignalAnalyzer(collector)
        result = await analyzer.analyze(tone_wav)
        assert result.sample_rate == 16000
