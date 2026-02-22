"""
EchoTrace TUI Application — the main Textual app root.

Acts purely as the application container and mounts Screens.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from textual.app import App

from echotrace.tui.screens.main import MainScreen


class EchoTraceApp(App):
    """The EchoTrace Terminal User Interface."""

    TITLE = "EchoTrace v0.1.0"
    CSS_PATH = "styles/app.tcss"

    BINDINGS = [
        ("r", "rerun", "Re-run"),
        ("e", "export", "Export JSON"),
        ("q", "quit", "Quit"),
        ("?", "help_screen", "Help"),
    ]

    def __init__(
        self,
        audio_path: str,
        reference_text: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_path = audio_path
        self.reference_text = reference_text

    def on_mount(self) -> None:
        """Mount the primary analysis screen."""
        self.push_screen(
            MainScreen(
                audio_path=self.audio_path,
                reference_text=self.reference_text,
                id="main-screen",
            )
        )

    def action_rerun(self) -> None:
        """Re-run analysis by calling the current screen's method."""
        screen = self.screen
        if isinstance(screen, MainScreen):
            screen.rerun()

    def action_export(self) -> None:
        """Export the report as JSON from the current screen."""
        screen = self.screen
        if isinstance(screen, MainScreen):
            report = screen.get_report()
            if report is None:
                self.notify("No report to export yet.", severity="warning")
                return

            export_data = report.to_export_dict()
            filename = f"echotrace_report_{report.job_id}.json"

            with open(filename, "w") as f:
                json.dump(export_data, f, indent=2)

            self.notify(f"Exported to {filename}", severity="information")

    def action_help_screen(self) -> None:
        """Show help information."""
        self.notify(
            "[R] Re-run analysis  |  [E] Export JSON  |  [Q] Quit",
            severity="information",
        )
