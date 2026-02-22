"""
EchoTrace CLI — the primary entry point.

Usage:
    echotrace                               → welcome screen
    echotrace demo                          → instant gratification
    echotrace analyze path/to/audio.wav     → run full pipeline
    echotrace setup                         → configure providers
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

from echotrace.presentation.cli_display import (
    console,
    display_welcome,
    display_error,
    display_demo_banner,
    display_file_not_found,
    display_unsupported_format,
    AnalysisProgressDisplay,
    render_scorecard,
    render_waterfall,
    render_transcript_diff,
    render_warnings,
    display_post_analysis_nudge,
    display_setup_header,
    display_setup_complete,
    display_first_run_check,
    display_gold_model_prompt,
    display_gold_model_alternatives,
)

# ── File-level log (never printed to terminal) ────────────────
_log_dir = Path.home() / ".config" / "echotrace"
_log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=_log_dir / "echotrace.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)

SUPPORTED_EXTS = {".wav", ".mp3", ".webm", ".m4a", ".flac", ".ogg"}

app = typer.Typer(
    name="echotrace",
    help=(
        "[bold cyan]EchoTrace[/] — The Voice AI Reliability Profiler.\n\n"
        "Chrome DevTools Network Tab, but for Audio."
    ),
    add_completion=False,
    rich_markup_mode="rich",
)


# ═════════════════════════════════════════════════════════════════
#  Callback — bare `echotrace` shows welcome screen
# ═════════════════════════════════════════════════════════════════

@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from echotrace import __version__
        display_welcome(version=__version__)
        raise typer.Exit()


# ═════════════════════════════════════════════════════════════════
#  analyze
# ═════════════════════════════════════════════════════════════════

@app.command(rich_help_panel="Core Analysis")
def analyze(
    audio_path: str = typer.Argument(
        ..., help="Path to the audio file to analyze."
    ),
    reference: str = typer.Option(
        None,
        "--reference",
        "-r",
        help='Reference transcript (gold standard). Example: "I want to book a heart checkup"',
    ),
    export_only: bool = typer.Option(
        False,
        "--export-only",
        help="Run analysis and export JSON without interactive display.",
    ),
) -> None:
    """
    Analyze a Voice AI audio file for latency, transcription errors,
    and signal quality.
    """
    # Suppress librosa / numba noise from terminal — only log to file
    logger.remove()
    logger.add(
        _log_dir / "echotrace.log",
        level="DEBUG",
        format="{time} | {level} | {message}",
        rotation="5 MB",
    )

    # ── Validate file ────────────────────────────────────────
    file_path = Path(audio_path)
    if not file_path.exists():
        display_file_not_found(audio_path)
        raise typer.Exit(code=1)

    if file_path.suffix.lower() not in SUPPORTED_EXTS:
        display_unsupported_format(file_path.suffix)
        raise typer.Exit(code=1)

    # ── First-run environment check (one-time) ───────────────
    from echotrace.config import load_state, save_state

    state = load_state()
    if not state.get("first_run_complete", False):
        _show_first_run_check(reference)
        state["first_run_complete"] = True
        save_state(state)

    # ── Gold model resolution (if no reference) ──────────────
    gold_model_override = None
    if not reference:
        gold_model_override = _resolve_gold_model()
        if gold_model_override is None:
            # User declined everything
            display_gold_model_alternatives()
            raise typer.Exit(code=0)

    # ── Run analysis ─────────────────────────────────────────
    if export_only:
        _run_export_only(str(file_path), reference, gold_model_override)
    else:
        _run_analysis(str(file_path), reference, is_demo=False, gold_model_override=gold_model_override)

    # Track state
    state = load_state()
    state["total_analyses"] = state.get("total_analyses", 0) + 1
    save_state(state)


# ═════════════════════════════════════════════════════════════════
#  demo
# ═════════════════════════════════════════════════════════════════

@app.command(rich_help_panel="Core Analysis")
def demo() -> None:
    """
    Run an instant demo analysis using a bundled audio file.
    No configuration or downloads required.
    """
    logger.remove()
    logger.add(
        _log_dir / "echotrace.log",
        level="DEBUG",
        format="{time} | {level} | {message}",
        rotation="5 MB",
    )

    demo_path = Path(__file__).parent / "data" / "demo.wav"
    if not demo_path.exists():
        display_error(
            "Bundled demo file not found",
            f"Expected at: {demo_path}",
            "Try reinstalling echotrace:",
            "pip install -e .",
        )
        raise typer.Exit(code=1)

    display_demo_banner()

    reference = "I'm going to cancel my credit card"
    _run_analysis(str(demo_path), reference, is_demo=True)

    # Track state
    from echotrace.config import load_state, save_state
    state = load_state()
    state["demo_run_count"] = state.get("demo_run_count", 0) + 1
    state["first_run_complete"] = True
    save_state(state)


# ═════════════════════════════════════════════════════════════════
#  setup
# ═════════════════════════════════════════════════════════════════

@app.command(rich_help_panel="Configuration")
def setup() -> None:
    """
    Launch an interactive wizard to configure EchoTrace providers.

    This command helps you set up your default environments, including:
    - Choosing between mock, local (Ollama), or cloud (Groq/HuggingFace) providers.
    - Setting default API keys (saved securely in ~/.config/echotrace/config.toml).
    - Overriding default LLM models used during Audio analysis.
    """
    from rich.prompt import Prompt, Confirm
    from echotrace.config import load_config, save_config, mark_setup_complete

    config = load_config()
    llm_conf = config.get("llm", {})
    current_provider = llm_conf.get("provider", "mock")

    # Check STT cache status
    stt_fast_available = _check_model_cached("tiny")
    stt_gold_cached = _check_model_cached("large-v2")

    display_setup_header(
        provider=current_provider,
        stt_fast_available=stt_fast_available,
        stt_gold_cached=stt_gold_cached,
    )

    # ── Step 1: LLM Provider ──────────────────────────────────
    console.print("  [bold]LLM Provider for TTFT measurement:[/]\n")
    console.print("  [bold cyan][1][/] mock         Simulated 800–2000ms delay. No setup required.")
    console.print("  [bold cyan][2][/] groq         Ultra-fast cloud inference. Free API key required.")
    console.print("  [bold cyan][3][/] ollama       Local inference. Requires Ollama running locally.")
    console.print("  [bold cyan][4][/] huggingface  Hosted inference API. Free API key required.\n")

    provider_map = {"1": "mock", "2": "groq", "3": "ollama", "4": "huggingface"}
    current_num = {v: k for k, v in provider_map.items()}.get(current_provider, "1")

    choice = Prompt.ask(f"  Choice [1-4]", default=current_num)
    provider = provider_map.get(choice, "mock")
    llm_conf["provider"] = provider
    provider_model = ""

    if provider == "groq":
        key = Prompt.ask("\n  Groq API Key (get free key at console.groq.com)", password=True,
                         default=llm_conf.get("groq_api_key", ""))
        llm_conf["groq_api_key"] = key
        # Test connection
        console.print("\n  Testing connection... ", end="")
        if _test_groq_connection(key):
            model = Prompt.ask("  Groq Model", default=llm_conf.get("groq_model", "llama3-8b-8192"))
            llm_conf["groq_model"] = model
            provider_model = model
            console.print(f"  [green]✓ connected ({model} available)[/]\n")
        else:
            console.print("  [red]✗ Connection failed[/]")
            if Confirm.ask("  Fall back to mock?", default=True):
                llm_conf["provider"] = "mock"
                provider = "mock"

    elif provider == "ollama":
        console.print("\n  Checking for Ollama at localhost:11434... ", end="")
        models = _detect_ollama()
        if models:
            console.print(f"[green]✓ Ollama detected[/]")
            console.print(f"  Available models: [cyan]{', '.join(models)}[/]\n")
            model = Prompt.ask("  Select model", default=llm_conf.get("ollama_model", "llama3.2:3b"))
            llm_conf["ollama_model"] = model
            provider_model = model
        else:
            console.print("[red]✗ Ollama not running[/]\n")
            console.print("  To use Ollama:")
            console.print("    1. Install from [link=https://ollama.ai]https://ollama.ai[/]")
            console.print("    2. Run: [cyan]ollama pull llama3.2:3b[/]")
            console.print("    3. Re-run: [cyan]echotrace setup[/]\n")
            if Confirm.ask("  Fall back to mock for now?", default=True):
                llm_conf["provider"] = "mock"
                provider = "mock"

    elif provider == "huggingface":
        key = Prompt.ask("\n  HuggingFace API Key", password=True,
                         default=llm_conf.get("hf_api_key", ""))
        llm_conf["hf_api_key"] = key

    # ── Step 2: Gold STT model ────────────────────────────────
    if not stt_gold_cached:
        console.print("\n  [bold]Gold Standard STT Model:[/]\n")
        console.print("  EchoTrace uses two Whisper models:")
        console.print("  · Fast model  (whisper-tiny,       ~75MB)  — simulates real-time STT")
        console.print("  · Gold model  (whisper-large-v2,  ~1.5GB) — reference for WER calculation\n")
        console.print("  whisper-tiny is already available.\n")

        if Confirm.ask("  Download whisper-large-v2 now? (~1.5GB, one-time download)", default=False):
            console.print("  [cyan]Downloading whisper-large-v2...[/]")
            _download_whisper_model("large-v2")
            stt_gold_cached = True
        else:
            console.print("  Skipped. EchoTrace will prompt on first [cyan]analyze[/] call.\n")

    config["llm"] = llm_conf
    save_config(config)
    mark_setup_complete()

    display_setup_complete(
        provider=provider,
        provider_model=provider_model,
        stt_gold_cached=stt_gold_cached,
    )


# ═════════════════════════════════════════════════════════════════
#  generate-fixtures
# ═════════════════════════════════════════════════════════════════

@app.command(rich_help_panel="Data & Fixtures")
def generate_fixtures(
    speech_dir: Path = typer.Option(
        ...,
        "--speech-dir",
        help="Directory containing clean speech audio files (.flac, .wav)",
    ),
    noise_dir: Optional[Path] = typer.Option(
        None,
        "--noise-dir",
        help="Directory containing noise audio files. If omitted, uses synthetic random noise.",
    ),
    out_dir: Path = typer.Option(
        Path("tests/fixtures/generated_dataset"),
        "--out-dir",
        help="Output directory for mixed audio and manifest",
    ),
    sample_rate: int = typer.Option(
        16000,
        "--sample-rate",
        help="Target sample rate (default 16000)",
    ),
    max_clips: int = typer.Option(
        10,
        "--max-clips",
        help="Maximum number of speech clips to process",
    ),
) -> None:
    """
    Generate synthetic test fixtures from local FLAC/WAV unmixed datasets.
    """
    from echotrace.fixtures.generator import generate_dataset

    try:
        generate_dataset(
            speech_dir=speech_dir,
            out_dir=out_dir,
            noise_dir=noise_dir,
            sample_rate=sample_rate,
            max_clips=max_clips,
        )
    except Exception as e:
        logging.exception("Fixture generation failure")
        display_error(
            "Failed to generate fixtures",
            str(e),
            "Check the log for details",
            "cat ~/.config/echotrace/echotrace.log",
        )
        raise typer.Exit(code=1)


# ═════════════════════════════════════════════════════════════════
#  version
# ═════════════════════════════════════════════════════════════════

@app.command(rich_help_panel="Utilities")
def version() -> None:
    """Print the EchoTrace version."""
    from echotrace import __version__
    typer.echo(f"EchoTrace v{__version__}")


# ═════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════

def _run_analysis(
    audio_path: str,
    reference: str | None,
    is_demo: bool = False,
    gold_model_override: str | None = None,
) -> None:
    """Run the full pipeline with live progress and rich report."""
    import asyncio
    from echotrace.core.orchestrator import Orchestrator

    async def _run():
        orchestrator = Orchestrator(
            audio_path=audio_path,
            reference_text=reference,
            gold_model_name=gold_model_override,
        )

        # Determine which stages to show
        stages = ["signal_analysis", "stt_tiny"]
        if not reference:
            stages.append("stt_gold")
        stages.append("llm_ttft")

        with AnalysisProgressDisplay(stages) as progress:
            def on_stage_complete(stage: str, duration_ms: float):
                progress.mark_done(stage, duration_ms)

            # Mark all stages as running initially
            for s in stages:
                progress.mark_running(s)

            report = await orchestrator.run(on_stage_complete=on_stage_complete)

        # Render results
        console.print()
        render_scorecard(
            score=report.reliability_score,
            snr_db=report.signal.snr_db,
            wer=report.transcription.wer,
            ttft_ms=report.llm.ttft_ms,
            e2e_ms=report.e2e_latency_ms,
            used_mock=report.llm.used_mock,
        )

        render_waterfall(report.collector.to_waterfall())

        if hasattr(report.transcription, 'diff') and report.transcription.diff:
            render_transcript_diff(
                diff_tokens=report.transcription.diff,
                wer=report.transcription.wer,
                gold_source=report.transcription.gold_source,
            )

        render_warnings(report.warnings)

        # Export JSON
        _export_report(report)

        display_post_analysis_nudge(report.job_id, is_demo=is_demo)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\n  [dim]Analysis cancelled.[/]\n")
    except Exception as e:
        logging.exception("Orchestrator failure")
        display_error(
            "Analysis failed",
            str(e),
            "Check the log for details",
            "cat ~/.config/echotrace/echotrace.log",
        )
        raise typer.Exit(code=1)


def _run_export_only(audio_path: str, reference: str | None, gold_model_override: str | None = None) -> None:
    """Run headless and just dump JSON to stdout."""
    import asyncio
    import json
    from echotrace.core.orchestrator import Orchestrator

    async def _run():
        orchestrator = Orchestrator(
            audio_path=audio_path,
            reference_text=reference,
            gold_model_name=gold_model_override,
        )
        report = await orchestrator.run()
        export = report.to_export_dict()
        print(json.dumps(export, indent=2))

    try:
        asyncio.run(_run())
    except Exception as e:
        logging.exception("Export failure")
        display_error("Export failed", str(e))
        raise typer.Exit(code=1)


def _export_report(report) -> None:
    """Save JSON report to disk."""
    import json
    export = report.to_export_dict()
    out_path = Path(f"echotrace_report_{report.job_id}.json")
    out_path.write_text(json.dumps(export, indent=2))


def _show_first_run_check(reference: str | None) -> None:
    """Display the one-time environment summary."""
    import platform

    checks = [
        ("ok", f"Python {platform.python_version()}", ""),
        ("ok", "faster-whisper installed", ""),
    ]

    if _check_model_cached("tiny"):
        checks.append(("ok", "whisper-tiny available", ""))
    else:
        checks.append(("warn", "whisper-tiny not cached", "— downloading on first use (~75MB)"))

    if _check_model_cached("large-v2"):
        checks.append(("ok", "whisper-large-v2 available", ""))
    else:
        checks.append(("warn", "whisper-large-v2 not cached", "— downloading on first use (~1.5GB)"))

    from echotrace.config import load_config
    config = load_config()
    provider = config.get("llm", {}).get("provider", "mock")
    if provider == "mock":
        checks.append(("info", f"LLM provider: mock", "(run 'echotrace setup' for real TTFT)"))
    else:
        checks.append(("ok", f"LLM provider: {provider}", ""))

    display_first_run_check(checks)


def _resolve_gold_model() -> str | None:
    """
    Resolve which gold model to use when no --reference is provided.
    Returns model name string, or None if user declines everything.
    """
    default_model = "large-v2"

    # If default is already cached, just use it silently
    if _check_model_cached(default_model):
        return default_model

    # Need to prompt
    size_str = _fetch_model_size(f"Systran/faster-whisper-{default_model}")
    if display_gold_model_prompt(f"Systran/faster-whisper-{default_model}", size_str):
        return default_model

    # User declined — offer alternatives
    from rich.prompt import Prompt
    console.print("\n  [cyan]Alternative models:[/]")
    console.print("  · Systran/faster-whisper-medium  (~1.5 GB)")
    console.print("  · Systran/faster-whisper-small   (~500 MB)\n")

    alt = Prompt.ask("  Enter a model ID (or press Enter to see other options)", default="")
    if alt.strip():
        return alt.strip()

    return None


def _check_model_cached(model_id: str) -> bool:
    """Check if a faster-whisper model exists in the HF cache."""
    if "/" not in model_id:
        repo_id = f"Systran/faster-whisper-{model_id}"
    else:
        repo_id = model_id
    cache_repo = "models--" + repo_id.replace("/", "--")
    cache_dir = Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub" / cache_repo
    return cache_dir.exists()


def _fetch_model_size(repo_id: str) -> str:
    """Fetch model size from HuggingFace API."""
    try:
        import requests
        r = requests.get(f"https://huggingface.co/api/models/{repo_id}", timeout=5)
        if r.status_code == 200:
            size_bytes = r.json().get("usedStorage", 0)
            if size_bytes > 0:
                size_mb = size_bytes / (1024 * 1024)
                if size_mb > 1024:
                    return f"~{size_mb / 1024:.1f} GB"
                return f"~{size_mb:.0f} MB"
    except Exception:
        pass
    return "~1.5 GB"


def _download_whisper_model(model_size: str) -> None:
    """Trigger a whisper model download by loading it."""
    try:
        from faster_whisper import WhisperModel
        WhisperModel(model_size, device="cpu", compute_type="int8")
        console.print(f"  [green]✓ whisper-{model_size} downloaded[/]\n")
    except Exception as e:
        logging.exception("Model download failure")
        display_error(f"Failed to download whisper-{model_size}", str(e))


def _test_groq_connection(api_key: str) -> bool:
    """Quick test to validate a Groq API key."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False


def _detect_ollama() -> list[str]:
    """Check if Ollama is running and return available model names."""
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            return [m["name"] for m in models]
    except Exception:
        pass
    return []


if __name__ == "__main__":
    app()
