"""
Orchestrator — coordinates the three async analysis tasks
using asyncio.gather() and feeds results to the Aggregator.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Callable, Optional

from loguru import logger

from echotrace.core.aggregator import AggregatedReport, Aggregator
from echotrace.analyzers.llm_probe import LLMProbe
from echotrace.analyzers.signal import SignalAnalyzer
from echotrace.analyzers.transcription import TranscriptionAuditor
from echotrace.config import EchoTraceConfig
from echotrace.providers import resolve_provider
from echotrace.core.telemetry import TelemetryCollector


class Orchestrator:
    """
    Spawns parallel async tasks and aggregates their results.

    Supports an optional on_stage_complete callback so the CLI
    can update a live progress display as each stage finishes.
    """

    def __init__(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        config: Optional[EchoTraceConfig] = None,
        gold_model_name: Optional[str] = None,
    ) -> None:
        self.audio_path = audio_path
        self.reference_text = reference_text
        self.job_id = uuid.uuid4().hex[:8]
        self.collector = TelemetryCollector()

        # Load config and resolve LLM provider
        self._config = config or EchoTraceConfig.load()
        self._provider = resolve_provider(self._config)

        logger.info(
            f"Orchestrator [{self.job_id}]: "
            f"LLM provider = {self._provider.label()}"
        )

        # Initialize analyzers
        self._signal = SignalAnalyzer(self.collector)
        self._transcription = TranscriptionAuditor(
            self.collector,
            reference_text=reference_text,
            gold_model_name=gold_model_name,
        )
        self._llm = LLMProbe(self.collector, provider=self._provider)

        # Results (populated after run)
        self.report: Optional[AggregatedReport] = None

    async def run(
        self,
        on_stage_complete: Optional[Callable[[str, float], None]] = None,
    ) -> AggregatedReport:
        """Execute the full analysis pipeline."""
        logger.info(
            f"Orchestrator [{self.job_id}]: Starting analysis for {self.audio_path}"
        )

        _callback = on_stage_complete or (lambda s, d: None)

        # --- Task A: Signal Analysis ---
        async def task_signal():
            result = await self._run_signal()
            duration = self.collector.stage_duration_ms("signal_analysis")
            _callback("signal_analysis", duration)
            return result

        # --- Task B: Transcription ---
        async def task_transcription():
            result = await self._run_transcription()
            # Fire callbacks for whichever stages ran
            tiny_dur = self.collector.stage_duration_ms("stt_tiny")
            _callback("stt_tiny", tiny_dur)
            gold_dur = self.collector.stage_duration_ms("stt_gold")
            if gold_dur > 0:
                _callback("stt_gold", gold_dur)
            return result

        # Run A and B in parallel
        signal_result, transcription_result = await asyncio.gather(
            task_signal(),
            task_transcription(),
        )

        # --- Task C: LLM Probe (depends on transcription text) ---
        gold_text = transcription_result.gold_transcript
        llm_result = await self._run_llm(gold_text)
        llm_dur = self.collector.stage_duration_ms("llm_ttft")
        _callback("llm_ttft", llm_dur)

        # --- Aggregate ---
        aggregator = Aggregator()
        self.report = aggregator.aggregate(
            job_id=self.job_id,
            file_path=self.audio_path,
            signal=signal_result,
            transcription=transcription_result,
            llm=llm_result,
            collector=self.collector,
        )

        logger.info(
            f"Orchestrator [{self.job_id}]: Complete. "
            f"Score={self.report.reliability_score}"
        )

        return self.report

    async def _run_signal(self):
        import time as _time
        start = _time.perf_counter()
        result = await self._signal.analyze(self.audio_path)
        end = _time.perf_counter()
        await self.collector.record("signal_analysis", start, end)
        return result

    async def _run_transcription(self):
        result = await self._transcription.audit(self.audio_path)
        return result

    async def _run_llm(self, prompt_text: str):
        import time as _time
        start = _time.perf_counter()
        result = await self._llm.probe(prompt_text)
        end = _time.perf_counter()
        await self.collector.record("llm_ttft", start, end)
        return result
