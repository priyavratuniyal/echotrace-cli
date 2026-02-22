"""
Transcript Diff Widget — shows the word-level diff between
gold standard and actual STT output with color coding.
"""

from __future__ import annotations

from typing import List, Optional

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, Static
from textual.reactive import reactive

from echotrace.analyzers.transcription import DiffToken


class TranscriptDiffWidget(Widget):
    """
    Renders a word-level diff between gold and actual transcripts.

    - Correct words: default color
    - Substitutions: [red] with strikethrough on original
    - Deletions: [red] (missing from actual)
    - Insertions: [green] (extra in actual)
    """

    CSS_PATH = "../styles/transcript_diff.tcss"

    diff_data = reactive(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Label("TRANSCRIPT DIFF", classes="section-title")
        yield Static("  Waiting for analysis...", id="diff-content")

    def watch_diff_data(self, data: Optional[dict]) -> None:
        self._render_diff()

    def _render_diff(self) -> None:
        content = self.query_one("#diff-content", Static)
        
        if not self.diff_data:
            return

        gold_text = self.diff_data.get("gold_text", "")
        actual_text = self.diff_data.get("actual_text", "")
        diff = self.diff_data.get("diff", [])
        gold_source = self.diff_data.get("gold_source", "")
        no_speech = self.diff_data.get("no_speech", False)

        if no_speech:
            content.update(
                "\n  [bold red]⚠ NO SPEECH DETECTED[/bold red]\n"
                "  The audio file contains no detectable speech.\n"
            )
            return

        lines: list[str] = []
        lines.append("")

        # Gold source label
        source_label = (
            "User Reference"
            if gold_source == "user-reference"
            else "Whisper Large-v2 (no reference provided)"
        )
        lines.append(f"  [dim]Gold standard: {source_label}[/dim]")
        lines.append("")

        # Gold transcript
        lines.append(f'  [bold]Gold:[/bold]   "{gold_text}"')

        # Actual transcript with diff markup
        actual_parts: list[str] = []
        for token in diff:
            if token.status == "correct":
                actual_parts.append(token.word)
            elif token.status == "substituted":
                actual_parts.append(f"[b #EAB308][{token.actual}][/]")
            elif token.status == "deleted":
                actual_parts.append(f"[s #EF4444]{token.word}[/]")
            elif token.status == "inserted":
                actual_parts.append(f"[b #B8F818]+{token.word}[/]")

        actual_display = " ".join(actual_parts)
        lines.append(f'  [bold]Actual:[/bold] "{actual_display}"')

        # Error summary
        subs = sum(1 for d in diff if d.status == "substituted")
        dels = sum(1 for d in diff if d.status == "deleted")
        ins = sum(1 for d in diff if d.status == "inserted")

        error_parts: list[str] = []
        if subs:
            error_parts.append(f"{subs} substitution{'s' if subs > 1 else ''}")
        if dels:
            error_parts.append(f"{dels} deletion{'s' if dels > 1 else ''}")
        if ins:
            error_parts.append(f"{ins} insertion{'s' if ins > 1 else ''}")

        if error_parts:
            lines.append("")
            lines.append(f"  {' | '.join(error_parts)}")

            # Detail individual errors
            for token in diff:
                if token.status == "substituted":
                    lines.append(
                        f'  Word: "{token.word}" → "{token.actual}"'
                    )
        else:
            lines.append("")
            lines.append("  [b #B8F818]✓ Perfect match — no errors detected[/]")

        lines.append("")
        content.update("\n".join(lines))
