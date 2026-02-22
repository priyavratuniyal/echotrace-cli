"""
Scorecard Widget — displays the reliability score,
signal metrics, and latency status in the top banner.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, LoadingIndicator, Static


class ScoreBox(Static):
    """The large central score display."""

    score = reactive(-1, recompose=True)  # -1 = loading
    status_label = reactive("ANALYZING...")

    def render(self) -> str:
        if self.score < 0:
            return "[dim]⏳[/dim]\n[dim]---[/dim]\n[dim]ANALYZING...[/dim]"

        # Determine status
        if self.score >= 80:
            status = "[b #B8F818]✓ GOOD[/]"
        elif self.score >= 50:
            status = "[b #EAB308]⚠ WARNING[/]"
        else:
            status = "[b #EF4444]✗ CRITICAL[/]"

        return f"[b]{self.score} / 100[/b]\n{status}"


class MetricLabel(Static):
    """A single metric with label, value, and severity indicator."""

    label_text = reactive("---", recompose=True)
    value_text = reactive("---", recompose=True)
    severity = reactive("neutral", recompose=True)

    def render(self) -> str:
        icon = {"good": "[#B8F818]✓[/]", "warning": "[#EAB308]⚠[/]", "critical": "[#EF4444]✗[/]", "neutral": "[dim]○[/dim]"}
        sev_icon = icon.get(self.severity, "[dim]○[/dim]")
        
        sev_color = {"good": "#B8F818", "warning": "#EAB308", "critical": "#EF4444", "neutral": "#6B7280"}.get(self.severity, "#6B7280")

        return f"[dim]{self.label_text}[/dim]\n[b]{self.value_text}[/b]\n{sev_icon} [{sev_color}]{self.severity.upper()}[/]"


class ScorecardWidget(Widget):
    """
    Top-level scorecard banner combining the score box
    and individual metric labels.
    """

    CSS_PATH = "../styles/scorecard.tcss"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._score_box = ScoreBox(id="score-box")
        
        self._snr_label = MetricLabel(id="snr-metric")
        self._snr_label.label_text = "SIGNAL QUALITY"
        
        self._wer_label = MetricLabel(id="wer-metric")
        self._wer_label.label_text = "TRANSCRIPTION"
        
        self._ttft_label = MetricLabel(id="ttft-metric")
        self._ttft_label.label_text = "LATENCY STATUS"
        
        self._p99_label = MetricLabel(id="p99-metric")
        self._p99_label.label_text = "P99 LATENCY"

    def compose(self) -> ComposeResult:
        with Horizontal(id="scorecard-row"):
            yield Vertical(
                Label("RELIABILITY SCORE", classes="section-title"),
                self._score_box,
                id="score-col",
            )
            yield Vertical(
                self._snr_label,
                self._wer_label,
                id="quality-col",
            )
            yield Vertical(
                self._ttft_label,
                self._p99_label,
                id="latency-col",
            )

    def set_score(self, score: int) -> None:
        self._score_box.score = score

    def set_signal(self, snr_db: float) -> None:
        if snr_db < 10:
            sev = "critical"
        elif snr_db < 20:
            sev = "warning"
        else:
            sev = "good"
        self._snr_label.value_text = f"SNR: {snr_db:.1f} dB"
        self._snr_label.severity = sev

    def set_transcription(self, wer: float | None) -> None:
        if wer is None:
            self._wer_label.value_text = "NO SPEECH"
            self._wer_label.severity = "critical"
            return
        pct = wer * 100
        if pct > 15:
            sev = "critical"
        elif pct > 5:
            sev = "warning"
        else:
            sev = "good"
        self._wer_label.value_text = f"WER: {pct:.1f}%"
        self._wer_label.severity = sev

    def set_ttft(self, ttft_ms: float, used_mock: bool) -> None:
        mock_tag = " [MOCK]" if used_mock else ""
        if ttft_ms > 2000:
            sev = "critical"
        elif ttft_ms > 1000:
            sev = "warning"
        else:
            sev = "good"
        self._ttft_label.value_text = f"TTFT: {ttft_ms:.0f}ms{mock_tag}"
        self._ttft_label.severity = sev

    def set_p99(self, p99_ms: float) -> None:
        if p99_ms > 2000:
            sev = "critical"
        elif p99_ms > 1000:
            sev = "warning"
        else:
            sev = "good"
        self._p99_label.value_text = f"P99: {p99_ms:.0f}ms"
        self._p99_label.severity = sev
