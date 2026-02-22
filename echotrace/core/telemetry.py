"""
Telemetry infrastructure for EchoTrace.

Every pipeline function is wrapped with @timed() to record
stage_name, start_time, end_time, and duration_ms into a
shared TelemetryCollector. This is the core data source
that powers the Latency Waterfall widget.
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class StageRecord:
    """A single recorded pipeline stage."""

    stage: str
    start_time: float
    end_time: float
    duration_ms: float
    metadata: dict = field(default_factory=dict)


@dataclass
class WaterfallBar:
    """Presentation-ready bar data for the TUI waterfall widget."""

    label: str
    duration_ms: float
    offset_ms: float  # relative to pipeline start
    color: str  # "green" | "yellow" | "red"
    flag: Optional[str] = None  # e.g. "← SLOW"


class TelemetryCollector:
    """
    Thread-safe accumulator for pipeline stage timings.

    Usage:
        collector = TelemetryCollector()

        @timed("stt_tiny", collector)
        async def transcribe(audio): ...
    """

    def __init__(self) -> None:
        self._stages: List[StageRecord] = []
        self._pipeline_start: Optional[float] = None
        self._lock = asyncio.Lock()

    async def record(
        self,
        stage: str,
        start: float,
        end: float,
        metadata: Optional[dict] = None,
    ) -> None:
        async with self._lock:
            if self._pipeline_start is None or start < self._pipeline_start:
                self._pipeline_start = start
            self._stages.append(
                StageRecord(
                    stage=stage,
                    start_time=start,
                    end_time=end,
                    duration_ms=round((end - start) * 1000, 2),
                    metadata=metadata or {},
                )
            )

    @property
    def stages(self) -> List[StageRecord]:
        return sorted(self._stages, key=lambda s: s.start_time)

    def to_waterfall(self) -> List[WaterfallBar]:
        """Convert raw stage records into presentation-ready bars."""
        bars: List[WaterfallBar] = []
        base = self._pipeline_start or 0.0

        for rec in self.stages:
            offset = round((rec.start_time - base) * 1000, 2)

            # Color thresholds
            if rec.duration_ms < 500:
                color = "green"
            elif rec.duration_ms < 1500:
                color = "yellow"
            else:
                color = "red"

            flag = None
            if rec.duration_ms > 1500:
                flag = "← SLOW"

            bars.append(
                WaterfallBar(
                    label=rec.stage,
                    duration_ms=rec.duration_ms,
                    offset_ms=offset,
                    color=color,
                    flag=flag,
                )
            )

        return bars

    @property
    def total_e2e_ms(self) -> float:
        if not self._stages:
            return 0.0
        earliest = min(s.start_time for s in self._stages)
        latest = max(s.end_time for s in self._stages)
        return round((latest - earliest) * 1000, 2)

    @property
    def p99_latency_ms(self) -> float:
        """Approximated from single run: max of all stage durations."""
        if not self._stages:
            return 0.0
        return max(s.duration_ms for s in self._stages)

    def reset(self) -> None:
        self._stages.clear()
        self._pipeline_start = None

    def stage_duration_ms(self, stage_name: str) -> float:
        """Return duration in ms for a named stage, or 0 if not recorded."""
        for s in self._stages:
            if s.stage == stage_name:
                return s.duration_ms
        return 0.0


def timed(stage_name: str, collector: Optional[TelemetryCollector] = None):
    """
    Decorator that records wall-clock execution time of an async function.

    Usage:
        collector = TelemetryCollector()

        @timed("signal_analysis", collector)
        async def analyze(path): ...

    If collector is None at decoration time, it can be passed at
    call time via a `_collector` kwarg (useful for dependency injection).
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Allow runtime injection of collector
            active_collector = kwargs.pop("_collector", None) or collector
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end = time.perf_counter()
                if active_collector is not None:
                    await active_collector.record(stage_name, start, end)

        return wrapper

    return decorator
