"""
Aggregator — computes reliability score and warnings from
the combined output of all three analysis tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from echotrace.analyzers.signal import SignalResult
from echotrace.analyzers.transcription import TranscriptionResult
from echotrace.analyzers.llm_probe import LLMProbeResult
from echotrace.core.telemetry import TelemetryCollector


@dataclass
class AggregatedReport:
    """Final report combining all pipeline outputs."""

    job_id: str
    file_path: str
    reliability_score: int
    signal: SignalResult
    transcription: TranscriptionResult
    llm: LLMProbeResult
    collector: TelemetryCollector
    warnings: List[str] = field(default_factory=list)

    @property
    def p99_latency_ms(self) -> float:
        return self.collector.p99_latency_ms

    @property
    def e2e_latency_ms(self) -> float:
        return self.collector.total_e2e_ms

    def to_export_dict(self) -> dict:
        """Produce the JSON export format."""
        from datetime import datetime, timezone

        waterfall = self.collector.to_waterfall()

        return {
            "job_id": self.job_id,
            "file": self.file_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reliability_score": self.reliability_score,
            "signal": {
                "snr_db": self.signal.snr_db,
                "silence_ratio": self.signal.silence_ratio,
                "duration_sec": self.signal.duration_sec,
                "vad_segments": [
                    {"start_ms": s.start_ms, "end_ms": s.end_ms}
                    for s in self.signal.vad_segments
                ],
            },
            "transcription": {
                "gold": self.transcription.gold_transcript.lower().strip(),
                "actual": self.transcription.tiny_transcript.lower().strip(),
                "wer": self.transcription.wer,
                "stt_latency_ms": self.transcription.stt_latency_ms,
                "gold_source": self.transcription.gold_source,
                "diff": [
                    {
                        "word": d.word,
                        "status": d.status,
                        **({"actual": d.actual} if d.actual else {}),
                    }
                    for d in self.transcription.diff
                ],
            },
            "latency": {
                "stages": [
                    {
                        "name": bar.label,
                        "duration_ms": bar.duration_ms,
                        "color": bar.color,
                    }
                    for bar in waterfall
                ],
                "e2e_ms": self.e2e_latency_ms,
                "p99_ms": self.p99_latency_ms,
                "used_mock_llm": self.llm.used_mock,
                "provider_name": self.llm.provider_name,
                "provider_model": self.llm.provider_model,
            },
            "warnings": self.warnings,
        }


class Aggregator:
    """Computes the final reliability score and generates warnings."""

    def aggregate(
        self,
        job_id: str,
        file_path: str,
        signal: SignalResult,
        transcription: TranscriptionResult,
        llm: LLMProbeResult,
        collector: TelemetryCollector,
    ) -> AggregatedReport:
        score = 100
        warnings: List[str] = []

        # --- Clip too short ---
        if signal.is_too_short:
            warnings.append(
                "CLIP_TOO_SHORT: Clip is < 1 second — analysis may be unreliable"
            )

        # --- No speech ---
        if transcription.no_speech:
            warnings.append(
                "NO_SPEECH: No speech detected in the audio file"
            )

        # --- WER penalties ---
        wer = transcription.wer
        if wer is not None:
            if wer > 0.15:
                score -= 30
                warnings.append(
                    f"CRITICAL_WER: Transcription accuracy critical "
                    f"({wer*100:.1f}% > 15% threshold)"
                )
            elif wer > 0.05:
                score -= 15
                warnings.append(
                    f"HIGH_WER: Transcription accuracy degraded "
                    f"({wer*100:.1f}% > 5% threshold)"
                )

        # --- SNR penalties ---
        if signal.snr_db < 10:
            score -= 25
            warnings.append(
                f"NOISY_SIGNAL: SNR critically low ({signal.snr_db}dB < 10dB)"
            )
        elif signal.snr_db < 20:
            score -= 10
            warnings.append(
                f"NOISY_SIGNAL: SNR below threshold ({signal.snr_db}dB < 20dB)"
            )

        # --- TTFT penalties ---
        if llm.ttft_ms > 2000:
            score -= 25
            warnings.append(
                f"CRITICAL_TTFT: LLM response latency critical "
                f"({llm.ttft_ms:.0f}ms > 2000ms threshold)"
            )
        elif llm.ttft_ms > 1000:
            score -= 10
            warnings.append(
                f"HIGH_TTFT: LLM response latency high "
                f"({llm.ttft_ms:.0f}ms > 1000ms threshold)"
            )

        # --- P99 pipeline latency penalties ---
        p99 = collector.p99_latency_ms
        if p99 > 5000:
            score -= 20
            warnings.append(
                f"CRITICAL_LATENCY: Pipeline P99 critically slow "
                f"({p99:.0f}ms > 5000ms threshold)"
            )
        elif p99 > 2000:
            score -= 10
            warnings.append(
                f"HIGH_LATENCY: Pipeline P99 above threshold "
                f"({p99:.0f}ms > 2000ms threshold)"
            )

        # --- STT latency penalties ---
        if transcription.stt_latency_ms > 30000:
            score -= 15
            warnings.append(
                f"SLOW_STT: Transcription latency critical "
                f"({transcription.stt_latency_ms:.0f}ms > 30s threshold)"
            )
        elif transcription.stt_latency_ms > 10000:
            score -= 5
            warnings.append(
                f"SLOW_STT: Transcription latency elevated "
                f"({transcription.stt_latency_ms:.0f}ms > 10s threshold)"
            )

        # --- Silence ratio ---
        if signal.silence_ratio > 0.4:
            score -= 10
            warnings.append(
                f"DEAD_AIR: Silence ratio too high "
                f"({signal.silence_ratio*100:.0f}% > 40%)"
            )

        score = max(0, min(100, score))

        return AggregatedReport(
            job_id=job_id,
            file_path=file_path,
            reliability_score=score,
            signal=signal,
            transcription=transcription,
            llm=llm,
            collector=collector,
            warnings=warnings,
        )
