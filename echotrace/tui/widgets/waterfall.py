"""
Waterfall Widget — renders the latency waterfall as
dynamically-sized horizontal bars in the terminal.
"""

from __future__ import annotations

from typing import List

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, Static
from textual.reactive import reactive

from echotrace.core.telemetry import WaterfallBar


class WaterfallRow(Static):
    """A single row in the waterfall: label + bar + duration."""

    def __init__(
        self,
        bar: WaterfallBar,
        max_duration: float,
        available_cols: int = 50,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bar = bar
        self._max_duration = max_duration
        self._available_cols = available_cols

    def render(self) -> str:
        bar = self._bar

        # Normalize bar width
        if self._max_duration > 0:
            ratio = bar.duration_ms / self._max_duration
        else:
            ratio = 0
        bar_width = max(1, int(ratio * self._available_cols))

        # Color mapping to Rich markup
        color_map = {"green": "#B8F818", "yellow": "#EAB308", "red": "#EF4444"}
        color = color_map.get(bar.color, "#6B7280")

        # Build bar characters
        bar_str = "█" * bar_width

        # Label padding
        label = f"[b]{bar.label:<14}[/b]"
        duration = f"[{color}]{bar.duration_ms:,.0f}ms[/{color}]"
        flag = f"  [dim]{bar.flag}[/dim]" if bar.flag else ""

        return f"  {label} [{color}]{bar_str}[/{color}] {duration}{flag}"


class WaterfallWidget(Widget):
    """
    Renders the full latency waterfall from TelemetryCollector data.
    """

    CSS_PATH = "../styles/waterfall.tcss"

    bars = reactive([])
    e2e_ms = reactive(0.0)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Label("LATENCY WATERFALL", classes="section-title")
        yield Static("  Waiting for analysis...", id="waterfall-content")

    def watch_bars(self, new_bars: List[WaterfallBar]) -> None:
        self._render_bars()

    def watch_e2e_ms(self, new_ms: float) -> None:
        self._render_bars()

    def _render_bars(self) -> None:
        content = self.query_one("#waterfall-content", Static)

        if not self.bars:
            content.update("  No timing data available.")
            return

        max_dur = max(b.duration_ms for b in self.bars) if self.bars else 1

        # Get terminal width estimate (conservative)
        try:
            available = self.size.width - 30  # leave room for labels + numbers
        except Exception:
            available = 50
        available = max(10, min(available, 80))

        lines: list[str] = []
        lines.append("")

        for bar in self.bars:
            ratio = bar.duration_ms / max_dur if max_dur > 0 else 0
            bar_width = max(1, int(ratio * available))

            color_map = {"green": "#B8F818", "yellow": "#EAB308", "red": "#EF4444"}
            color = color_map.get(bar.color, "#6B7280")

            bar_str = "█" * bar_width
            label = f"[b]{bar.label:<14}[/b]"
            duration = f"[{color}]{bar.duration_ms:,.0f}ms[/{color}]"
            flag = f"  [dim]{bar.flag}[/dim]" if bar.flag else ""

            lines.append(
                f"  {label} [{color}]{bar_str}[/{color}] {duration}{flag}"
            )

        lines.append("")
        lines.append(f"  [b]Total E2E:[/b] [#B8F818]{self.e2e_ms:,.0f}ms[/#B8F818]")
        lines.append("")

        content.update("\n".join(lines))
