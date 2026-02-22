"""
cli_display.py — Centralized Rich rendering for EchoTrace.

ALL terminal output goes through this module. No other file in
the project should create Console instances or call rich.print.

Every function accepts pure data (dataclasses, dicts, primitives)
and contains ZERO business logic.
"""

from __future__ import annotations

from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.text import Text
from rich.live import Live
from rich.prompt import Confirm, Prompt
from rich import box

# ── Module-level singletons ──────────────────────────────────────
console = Console()
error_console = Console(stderr=True, style="bold red")


# ═════════════════════════════════════════════════════════════════
#  Screen 1: Welcome
# ═════════════════════════════════════════════════════════════════

def display_welcome(version: str = "0.1.0") -> None:
    """Show the welcome panel when `echotrace` is run with no subcommand."""
    content = Text()
    content.append("\n")
    content.append("  New here? Start with the demo — no setup required:\n\n", style="dim")
    content.append("    $ echotrace demo\n\n", style="bold cyan")
    content.append("  Analyze your own audio:\n\n", style="dim")
    content.append("    $ echotrace analyze <path/to/audio.wav>\n\n", style="bold cyan")
    content.append("  Set up API integrations (Groq, Ollama, HuggingFace):\n\n", style="dim")
    content.append("    $ echotrace setup\n\n", style="bold cyan")
    content.append("  Generate test fixtures from your own audio:\n\n", style="dim")
    content.append("    $ echotrace generate-fixtures --help\n\n", style="bold cyan")

    header = Text()
    header.append(f"  ▓▓ EchoTrace v{version}", style="bold white")
    header.append("  ·  Voice AI Reliability Profiler\n", style="bold white")
    header.append('  "The Chrome DevTools Network Tab, but for Audio."', style="dim")

    console.print(Panel(
        Text.assemble(header, "\n", content),
        box=box.ROUNDED,
        border_style="cyan",
        padding=(0, 1),
    ))
    console.print("  Run  [bold cyan]echotrace <command> --help[/]  for detailed usage.\n", highlight=False)


# ═════════════════════════════════════════════════════════════════
#  Error Display
# ═════════════════════════════════════════════════════════════════

def display_error(
    title: str,
    detail: str = "",
    fix: str | None = None,
    command: str | None = None,
) -> None:
    """Structured error output — never let raw tracebacks hit the user."""
    console.print()
    console.print(f"  [bold red]✗ {title}[/]")
    if detail:
        console.print(f"\n    {detail}")
    if fix:
        console.print(f"\n    {fix}")
    if command:
        console.print(f"      [bold cyan]$ {command}[/]")
    console.print()


# ═════════════════════════════════════════════════════════════════
#  Screen 2a: Demo Banner
# ═════════════════════════════════════════════════════════════════

def display_demo_banner(
    filename: str = "demo.wav",
    duration: str = "3.2s",
    scenario: str = "Voice order at a noisy café counter",
    transcript: str = "I'm going to cancel my credit card",
) -> None:
    """Print the demo intro card before analysis starts."""
    console.print()
    console.print(Rule(" EchoTrace Demo ", style="bold cyan"))
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="dim")
    info_table.add_column(style="white")
    info_table.add_row("Sample file", f"[bold]{filename}[/]  (built-in, {duration})")
    info_table.add_row("Scenario", scenario)
    info_table.add_row("Transcript", f'[italic]"{transcript}"[/]')
    info_table.add_row("", "")
    info_table.add_row("Showcases", "latency waterfall, WER diff, SNR scoring")
    console.print(info_table)
    console.print(Rule(style="dim"))
    console.print()


# ═════════════════════════════════════════════════════════════════
#  Screen 2b: Live Analysis Progress
# ═════════════════════════════════════════════════════════════════

_STAGE_LABELS = {
    "signal_analysis": "Signal Analysis",
    "stt_tiny": "STT Audit",
    "stt_gold": "STT Gold Reference",
    "llm_ttft": "LLM Probe",
}


class AnalysisProgressDisplay:
    """
    Manages a rich.Live display showing per-stage progress.

    Usage:
        progress = AnalysisProgressDisplay(stages)
        with progress:
            # ... orchestrator runs ...
            progress.mark_done("signal_analysis", 12.3)
    """

    def __init__(self, stages: list[str]) -> None:
        self._stages = stages
        self._status: dict[str, str] = {s: "pending" for s in stages}
        self._durations: dict[str, float] = {}
        self._live = Live(self._build_table(), console=console, refresh_per_second=8)

    def __enter__(self):
        self._live.__enter__()
        return self

    def __exit__(self, *args):
        # Final render with all done
        self._live.update(self._build_table())
        self._live.__exit__(*args)

    def mark_running(self, stage: str) -> None:
        self._status[stage] = "running"
        self._live.update(self._build_table())

    def mark_done(self, stage: str, duration_ms: float) -> None:
        self._status[stage] = "done"
        self._durations[stage] = duration_ms
        self._live.update(self._build_table())

    def mark_failed(self, stage: str) -> None:
        self._status[stage] = "failed"
        self._live.update(self._build_table())

    def _build_table(self) -> Table:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(width=3)
        table.add_column(min_width=20)
        table.add_column(justify="right", min_width=12)

        for stage in self._stages:
            label = _STAGE_LABELS.get(stage, stage)
            status = self._status[stage]
            if status == "pending":
                icon = "[dim]◌[/]"
                status_text = "[dim]pending[/]"
            elif status == "running":
                icon = "[cyan]⠋[/]"
                status_text = "[cyan]running...[/]"
            elif status == "done":
                dur = self._durations.get(stage, 0)
                icon = "[green]●[/]"
                status_text = f"[green]done[/]  ({dur:,.0f}ms)"
            else:
                icon = "[red]✗[/]"
                status_text = "[red]failed[/]"

            table.add_row(icon, label, status_text)

        return table


# ═════════════════════════════════════════════════════════════════
#  Screen 2c — Panel 1: Reliability Scorecard
# ═════════════════════════════════════════════════════════════════

def render_scorecard(
    score: int,
    snr_db: float,
    wer: float | None,
    ttft_ms: float,
    e2e_ms: float,
    used_mock: bool,
) -> None:
    """Print the reliability scorecard panel."""
    # Score styling
    if score >= 80:
        score_style, status = "bold green", "HEALTHY"
    elif score >= 50:
        score_style, status = "bold yellow", "WARNING"
    else:
        score_style, status = "bold red", "CRITICAL"

    # SNR
    snr_style = "green" if snr_db >= 20 else ("yellow" if snr_db >= 10 else "red")
    snr_note = "Clean signal" if snr_db >= 20 else ("Noisy signal" if snr_db >= 10 else "Unusable signal")

    # WER
    if wer is not None:
        wer_pct = wer * 100
        wer_style = "green" if wer_pct <= 5 else ("yellow" if wer_pct <= 15 else "red")
        wer_note = "Accurate" if wer_pct <= 5 else ("Degraded accuracy" if wer_pct <= 15 else "Critical accuracy")
    else:
        wer_pct = None
        wer_style = "dim"
        wer_note = "No speech"

    # TTFT
    ttft_style = "green" if ttft_ms <= 1000 else ("yellow" if ttft_ms <= 2000 else "red")
    ttft_note = "Good" if ttft_ms <= 1000 else ("High latency" if ttft_ms <= 2000 else "Critical latency")

    content = Text()
    content.append(f"\n   Score    ", style="dim")
    content.append(f"{score} / 100", style=score_style)
    content.append(f"   {status}\n", style=score_style)
    content.append("   ──────────────────────────────────────────────────────\n", style="dim")
    content.append(f"   SNR   ", style="dim")
    content.append(f"{snr_db:>10.1f} dB", style=snr_style)
    content.append(f"     {snr_note}\n", style=snr_style)
    content.append(f"   WER   ", style="dim")
    if wer_pct is not None:
        content.append(f"{wer_pct:>10.1f} %", style=wer_style)
        content.append(f"     {wer_note}\n", style=wer_style)
    else:
        content.append(f"       N/A", style="dim")
        content.append(f"     {wer_note}\n", style="dim")
    content.append(f"   TTFT  ", style="dim")
    content.append(f"{ttft_ms:>10,.0f} ms", style=ttft_style)
    content.append(f"     {ttft_note}", style=ttft_style)
    if used_mock:
        content.append(f"  [mock]", style="dim")
    content.append(f"\n", style="dim")
    content.append(f"   E2E   ", style="dim")
    content.append(f"{e2e_ms:>10,.0f} ms", style="white")
    content.append(f"     total\n", style="dim")

    console.print(Panel(content, title="Reliability Scorecard", box=box.ROUNDED, border_style="white"))


# ═════════════════════════════════════════════════════════════════
#  Screen 2c — Panel 2: Latency Waterfall
# ═════════════════════════════════════════════════════════════════

def render_waterfall(bars: list, console_width: int | None = None) -> None:
    """
    Print a horizontal bar-chart of pipeline stage durations.

    bars: list of objects with .label, .duration_ms, .color, .flag attributes
    """
    if not bars:
        return

    width = (console_width or console.width) - 30
    width = max(width, 10)
    max_dur = max(b.duration_ms for b in bars)
    max_dur = max(max_dur, 1)

    # Identify the single slowest stage
    slowest_idx = max(range(len(bars)), key=lambda i: bars[i].duration_ms)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", min_width=18)
    table.add_column()
    table.add_column(justify="right")

    for i, b in enumerate(bars):
        bar_w = max(1, int((b.duration_ms / max_dur) * width))

        # Color by duration
        if b.duration_ms < 500:
            color = "green"
        elif b.duration_ms < 1500:
            color = "yellow"
        else:
            color = "red"

        bar_str = f"[{color}]{'▓' * bar_w}[/]"

        dur_str = f"[{color}]{b.duration_ms:,.0f}ms[/]"
        if i == slowest_idx and b.duration_ms > 1000:
            dur_str += " [bold red]◀ SLOW[/]"

        table.add_row(b.label, bar_str, dur_str)

    total_e2e = sum(b.duration_ms for b in bars)
    table.add_row("", "", "")
    table.add_row("", "", f"[dim]Total E2E: {total_e2e:,.0f}ms[/]")

    console.print(Panel(table, title="Latency Waterfall", box=box.ROUNDED, border_style="white"))


# ═════════════════════════════════════════════════════════════════
#  Screen 2c — Panel 3: Transcript Diff
# ═════════════════════════════════════════════════════════════════

def render_transcript_diff(
    diff_tokens: list,
    wer: float | None,
    gold_source: str,
) -> None:
    """
    Print a word-level diff panel.

    diff_tokens: list of DiffToken objects with .word, .status, .actual attributes
    """
    if not diff_tokens:
        return

    gold_display = Text("  Gold    ")
    actual_display = Text("  Actual  ")

    subs, dels, inss = 0, 0, 0

    for tk in diff_tokens:
        if tk.status == "correct":
            gold_display.append(f"{tk.word} ", style="white")
            actual_display.append(f"{tk.word} ", style="white")
        elif tk.status == "substituted":
            subs += 1
            gold_display.append(f"[{tk.word}]", style="bold red")
            gold_display.append(" ")
            actual_display.append(f"[{tk.actual}]", style="bold red")
            actual_display.append(" ")
        elif tk.status == "deleted":
            dels += 1
            gold_display.append(tk.word + " ", style="red strikethrough")
        elif tk.status == "inserted":
            inss += 1
            actual_display.append(tk.word + " ", style="bold green")

    # Summary line
    parts = []
    if subs > 0:
        parts.append(f"{subs} substitution{'s' if subs != 1 else ''}")
    if dels > 0:
        parts.append(f"{dels} deletion{'s' if dels != 1 else ''}")
    if inss > 0:
        parts.append(f"{inss} insertion{'s' if inss != 1 else ''}")

    summary = "  ·  ".join(parts)
    if wer is not None:
        summary += f"  ·  WER: {wer * 100:.1f}%"

    content = Text()
    content.append(f"\n  Gold standard: {gold_source}\n\n", style="dim")
    content = Text.assemble(content, gold_display, "\n", actual_display)
    if summary:
        content = Text.assemble(content, "\n\n  ", Text(summary, style="dim"), "\n")

    console.print(Panel(content, title="Transcript Diff", box=box.ROUNDED, border_style="white"))


# ═════════════════════════════════════════════════════════════════
#  Screen 2c — Panel 4: Warnings
# ═════════════════════════════════════════════════════════════════

def render_warnings(warnings: list[str]) -> None:
    """Print warnings panel, or a green healthy line if none."""
    if not warnings:
        console.print("  [bold green]✓ No issues detected — pipeline is healthy[/]\n")
        return

    lines = []
    for w in warnings:
        # Split on first colon to get tag and description
        if ":" in w:
            tag, desc = w.split(":", 1)
            lines.append(f"  ⚠  [bold yellow]{tag.strip()}[/]  {desc.strip()}")
        else:
            lines.append(f"  ⚠  {w}")

    content = "\n".join(lines)
    console.print(Panel(content, title="Warnings", box=box.ROUNDED, border_style="yellow"))


# ═════════════════════════════════════════════════════════════════
#  Screen 2d: Post-analysis nudge
# ═════════════════════════════════════════════════════════════════

def display_post_analysis_nudge(job_id: str, is_demo: bool = False) -> None:
    """Print next-step CTAs after results render."""
    console.print(Rule(style="dim"))
    console.print(f"  Report saved: [bold]echotrace_report_{job_id}.json[/]\n")
    if is_demo:
        console.print("  Try it on your own audio:")
        console.print("    [bold cyan]$ echotrace analyze path/to/your/audio.wav[/]\n")
        console.print("  Unlock real LLM latency measurement:")
        console.print("    [bold cyan]$ echotrace setup[/]")
    else:
        console.print("  Configure LLM provider for real TTFT measurement:")
        console.print("    [bold cyan]$ echotrace setup[/]")
    console.print(Rule(style="dim"))
    console.print()


# ═════════════════════════════════════════════════════════════════
#  Screen 3a: Setup header (current config state)
# ═════════════════════════════════════════════════════════════════

def display_setup_header(
    provider: str,
    stt_fast_available: bool,
    stt_gold_cached: bool,
    gold_model: str = "large-v2",
) -> None:
    """Show the current config state at the top of the setup wizard."""
    console.print()
    console.print(Rule(" EchoTrace Setup ", style="bold cyan"))

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=16)
    table.add_column()

    provider_note = "[dim]← no API keys configured[/]" if provider == "mock" else "[green]✓[/]"
    table.add_row("LLM Provider", f"[bold]{provider}[/]  {provider_note}")

    fast_icon = "[green]✓ available[/]" if stt_fast_available else "[red]✗ not cached[/]"
    table.add_row("STT Fast", f"whisper-tiny  {fast_icon}")

    gold_icon = "[green]✓ cached[/]" if stt_gold_cached else f"[yellow]✗ not cached[/]"
    table.add_row("STT Gold", f"whisper-{gold_model}  {gold_icon}")

    console.print(Panel(table, title="Current Configuration", subtitle="~/.config/echotrace/config.toml", box=box.ROUNDED, border_style="dim"))
    console.print("  Let's configure your integrations.\n")


# ═════════════════════════════════════════════════════════════════
#  Screen 3d: Setup complete
# ═════════════════════════════════════════════════════════════════

def display_setup_complete(
    provider: str,
    provider_model: str = "",
    stt_fast: str = "whisper-tiny",
    stt_gold_cached: bool = False,
    gold_model: str = "large-v2",
) -> None:
    """Print the final confirmation after setup wizard completes."""
    console.print(Rule(style="dim"))
    console.print("  [bold green]Configuration saved[/] to [dim]~/.config/echotrace/config.toml[/]\n")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="dim", min_width=16)
    table.add_column()

    prov_detail = f"{provider}  [green]✓[/]"
    if provider_model:
        prov_detail += f"  ({provider_model})"
    table.add_row("LLM Provider", prov_detail)
    table.add_row("STT Fast", f"{stt_fast}  [green]✓[/]")

    gold_icon = "[green]✓ cached[/]" if stt_gold_cached else "[yellow]⚠ download on first use[/]"
    table.add_row("STT Gold", f"whisper-{gold_model}  {gold_icon}")

    console.print(table)
    console.print("\n  You're ready. Try a real analysis:")
    console.print("    [bold cyan]$ echotrace analyze your_audio.wav[/]")
    console.print(Rule(style="dim"))
    console.print()


# ═════════════════════════════════════════════════════════════════
#  Screen 4: First-run environment check
# ═════════════════════════════════════════════════════════════════

def display_first_run_check(checks: list[tuple[str, str, str]]) -> None:
    """
    One-time environment summary.

    checks: list of (icon, label, detail) tuples
        icon: "ok" | "warn" | "info"
    """
    console.print("\n  [bold]First run — checking environment...[/]\n")
    for icon_type, label, detail in checks:
        if icon_type == "ok":
            icon = "[green]✓[/]"
        elif icon_type == "warn":
            icon = "[yellow]⚠[/]"
        else:
            icon = "[dim]○[/]"
        console.print(f"  {icon} {label}  {detail}")
    console.print(Rule(style="dim"))
    console.print()


# ═════════════════════════════════════════════════════════════════
#  File validation errors (Screen 4 variants)
# ═════════════════════════════════════════════════════════════════

SUPPORTED_FORMATS = {"WAV", "MP3", "FLAC", "OGG", "M4A", "WEBM"}
SUPPORTED_STR = " · ".join(sorted(SUPPORTED_FORMATS))

def display_file_not_found(path: str) -> None:
    display_error(
        f"File not found: {path}",
        "Make sure the path is correct and the file exists.",
        f"Supported formats: {SUPPORTED_STR}",
    )
    console.print("  Don't have an audio file? Try the built-in demo:")
    console.print("    [bold cyan]$ echotrace demo[/]\n")


def display_unsupported_format(suffix: str) -> None:
    display_error(
        f"Unsupported format: {suffix}",
        f"Supported formats: {SUPPORTED_STR}",
        "Convert with ffmpeg:",
        f"ffmpeg -i your_file{suffix} converted.wav",
    )


def display_audio_too_short(duration: float) -> None:
    display_error(
        f"Audio too short: {duration:.1f} seconds (minimum: 1.0 second)",
        "EchoTrace needs at least 1 second of audio for reliable analysis.",
    )


# ═════════════════════════════════════════════════════════════════
#  Gold model download prompt (Screen 5)
# ═════════════════════════════════════════════════════════════════

def display_gold_model_prompt(model_name: str, size_str: str) -> bool:
    """
    Ask user for consent to download gold STT model.
    Returns True if user agrees.
    """
    console.print("\n  EchoTrace needs a [bold]Gold Standard STT model[/] to calculate WER.\n")
    console.print("  No reference text provided ([dim]--reference flag not used[/]).")
    console.print(f"  Recommended model: [bold cyan]{model_name}[/]\n")

    table = Table(show_header=False, box=box.ROUNDED, padding=(0, 2))
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Model", model_name)
    table.add_row("Size", size_str)
    table.add_row("Purpose", "Used as accuracy baseline (one-time download)")
    console.print(table)
    console.print()

    return Confirm.ask("  Download now?", default=True)


def display_gold_model_alternatives() -> None:
    """Show alternative approaches when user declines gold model download."""
    console.print("\n  [bold]Alternatives:[/]\n")
    console.print("  [bold cyan][1][/]  Provide reference text inline:")
    console.print('      [dim]$ echotrace analyze audio.wav --reference "exact words spoken"[/]\n')
    console.print("  [bold cyan][2][/]  Use a smaller model (less accurate baseline):")
    console.print("      [dim]$ echotrace analyze audio.wav[/]  → then choose a smaller model\n")
    console.print("  [bold cyan][3][/]  Skip WER — latency analysis only:")
    console.print("      [dim]$ echotrace analyze audio.wav --no-wer[/]\n")
