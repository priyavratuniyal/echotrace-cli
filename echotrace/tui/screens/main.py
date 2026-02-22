"""
Main Screen — handles the layout grid and background analysis worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual import work, on
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.screen import Screen
from textual.widgets import Footer, Header, Static

from echotrace.core.aggregator import AggregatedReport
from echotrace.core.orchestrator import Orchestrator
from echotrace.tui.widgets.scorecard import ScorecardWidget
from echotrace.tui.widgets.transcript_diff import TranscriptDiffWidget
from echotrace.tui.widgets.waterfall import WaterfallWidget


class MainScreen(Screen):
    """The primary dashboard screen that runs analysis."""

    class AnalysisComplete(Message):
        def __init__(self, report: AggregatedReport) -> None:
            self.report = report
            super().__init__()

    class AnalysisFailed(Message):
        def __init__(self, error: str) -> None:
            self.error = error
            super().__init__()

    def __init__(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_path = audio_path
        self.reference_text = reference_text
        self._running = False
        self._report: Optional[AggregatedReport] = None

    def compose(self) -> ComposeResult:
        yield Header()

        with VerticalScroll(id="main-container"):
            filename = Path(self.audio_path).name
            yield Static(
                f"  [b]file:[/b] {filename}  |  "
                f"[b]status:[/b] [yellow]analyzing...[/yellow]",
                id="header-bar",
            )
            yield ScorecardWidget(id="scorecard")
            yield WaterfallWidget(id="waterfall")
            yield TranscriptDiffWidget(id="transcript-diff")
            yield Static("", id="warnings-box")

        yield Footer()

    def on_mount(self) -> None:
        """Start the analysis pipeline when mounted."""
        self.run_analysis()

    @work(exclusive=True)
    async def run_analysis(self) -> None:
        """Background worker that calls the core domain logic."""
        self._running = True
        
        try:
            orchestrator = Orchestrator(
                audio_path=self.audio_path,
                reference_text=self.reference_text,
            )
            
            # Update header with job ID proactively using call_from_thread
            filename = Path(self.audio_path).name
            header = self.query_one("#header-bar", Static)
            self.call_from_thread(
                header.update,
                f"  [b]file:[/b] {filename}  |  "
                f"[b]job:[/b] {orchestrator.job_id}  |  "
                f"[b]status:[/b] [yellow]analyzing...[/yellow]"
            )

            report = await orchestrator.run()
            self.post_message(self.AnalysisComplete(report))
        except Exception as e:
            self.post_message(self.AnalysisFailed(str(e)))

    @on(AnalysisComplete)
    def handle_analysis_complete(self, message: AnalysisComplete) -> None:
        """Handles the completion message from the background worker."""
        self._running = False
        self._report = message.report
        report = message.report

        scorecard = self.query_one("#scorecard", ScorecardWidget)
        waterfall = self.query_one("#waterfall", WaterfallWidget)
        diff_widget = self.query_one("#transcript-diff", TranscriptDiffWidget)
        header = self.query_one("#header-bar", Static)
        warnings_box = self.query_one("#warnings-box", Static)

        # Update Reactive Properties on Widgets
        scorecard.set_score(report.reliability_score)
        scorecard.set_signal(report.signal.snr_db)
        scorecard.set_transcription(report.transcription.wer)
        scorecard.set_ttft(report.llm.ttft_ms, report.llm.used_mock)
        scorecard.set_p99(report.p99_latency_ms)

        waterfall.e2e_ms = report.e2e_latency_ms
        waterfall.bars = report.collector.to_waterfall()

        diff_widget.diff_data = {
            "gold_text": report.transcription.gold_transcript,
            "actual_text": report.transcription.tiny_transcript,
            "diff": report.transcription.diff,
            "gold_source": report.transcription.gold_source,
            "no_speech": report.transcription.no_speech
        }

        # Update pure Statics
        filename = Path(self.audio_path).name
        header.update(
            f"  [b]file:[/b] {filename}  |  "
            f"[b]job:[/b] {report.job_id}  |  "
            f"[b]status:[/b] [b #B8F818]complete ✓[/]"
        )

        if report.warnings:
            warn_lines = ["  [b]⚠ WARNINGS[/b]", ""]
            for w in report.warnings:
                warn_lines.append(f"  • {w}")
            warn_lines.append("")
            warnings_box.update("\n".join(warn_lines))
        else:
            warnings_box.update("  [b #B8F818]✓ No warnings — all metrics within thresholds[/]")

    @on(AnalysisFailed)
    def handle_analysis_failed(self, message: AnalysisFailed) -> None:
        """Handles failure from the background worker."""
        self._running = False
        header = self.query_one("#header-bar", Static)
        filename = Path(self.audio_path).name
        header.update(
            f"  [b]file:[/b] {filename}  |  "
            f"[b #EF4444]ERROR: {message.error}[/]"
        )

    def rerun(self) -> None:
        if not self._running:
            header = self.query_one("#header-bar", Static)
            filename = Path(self.audio_path).name
            header.update(
                f"  [b]file:[/b] {filename}  |  "
                f"[b]status:[/b] [yellow]re-analyzing...[/yellow]"
            )
            self.run_analysis()

    def get_report(self) -> Optional[AggregatedReport]:
        return self._report
