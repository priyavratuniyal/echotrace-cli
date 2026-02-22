# 🎙️ EchoTrace: The Voice AI Reliability Profiler

[![CI](https://github.com/priyavratuniyal/echotrace-cli/actions/workflows/ci.yml/badge.svg)](https://github.com/priyavratuniyal/echotrace-cli/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/echotrace.svg)](https://badge.fury.io/py/echotrace)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/echotrace.svg)](https://pypi.org/project/echotrace/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/echotrace)](https://pepy.tech/project/echotrace)

**Chrome DevTools Network Tab, but for Audio.**
EchoTrace analyzes Voice AI audio logs and renders a latency waterfall in your terminal, allowing engineers to identify precisely which pipeline stage is creating bottlenecks or accuracy degradation.

Built specifically for Voice AI engineers debugging slow or inaccurate STT -> LLM -> TTS pipelines.

## 🛠️ Installation & Setup

### From PyPI (Recommended)

```bash
pip install echotrace
```

### From Source

For development or to get the latest unreleased changes:

```bash
# 1. Clone the repository
git clone https://github.com/priyavratuniyal/echotrace-cli.git
cd echotrace-cli

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# 3. Install the package in editable mode
pip install -e .
```

### ⚙️ Configuration Setup

Run the interactive setup wizard to configure your preferred metrics providers (Local/Cloud).

```bash
echotrace setup
```
This wizard helps you specify your backend for evaluating metrics like Time To First Token (TTFT). You have the option to use Cloud API keys (e.g., Groq) or local inference with Ollama.

#### Using Ollama (Local LLM)
If you want completely local execution without sending data to third-parties, download [Ollama](https://ollama.com/) and follow these steps before running `echotrace setup`:

```bash
# Start your local ollama server
ollama serve

# Keep it running, and open a new terminal tab to pull your model (e.g., Llama 3)
ollama pull llama3

# Now run 'echotrace setup' and select 'ollama' when prompted!
```

## ⚡ Quick Start

Once installed and configured, you are ready to profile audio pipelines!

```bash
# Analyze a single audio file (launches interactive TUI)
echotrace analyze .extras/output/mixed/demo_voice_1_mixed_with_noise_1_snr10.wav

# Analyze with a reference gold-standard transcript to precisely calculate WER
echotrace analyze .extras/output/mixed/demo_voice_1_mixed_with_noise_1_snr10.wav --reference "I am going to cancel my credit card"

# Headless mode (exports the complete analysis JSON block to stdout)
echotrace analyze .extras/output/mixed/demo_voice_1_mixed_with_noise_1_snr10.wav --export-only
```

## Core Features and Interface

EchoTrace provides an interactive terminal user interface (TUI) that surfaces the most critical metrics for conversational AI pipelines:

```text
┌──────────────────────────────────────────────────────────────┐
│  EchoTrace v0.1.0  |  job: a3f1b2c4  |  status: complete     │
├──────────────────────────────────────────────────────────────┤
│  RELIABILITY SCORE        SIGNAL QUALITY    LATENCY STATUS   │
│  ┌──────────────┐         SNR: 14.2 dB      P99: 2,340ms    │
│  │   72 / 100   │         NOISY             CRITICAL        │
│  │   WARNING    │         WER: 8.3%         TTFT: 1,840ms   │
│  └──────────────┘         DEGRADED          HIGH            │
├──────────────────────────────────────────────────────────────┤
│  LATENCY WATERFALL                                           │
│  signal_analysis ████ 120ms                                  │
│  stt_audit       ████████████████ 740ms                      │
│  llm_ttft        ████████████████████████████ 1,840ms <- SLOW│
├──────────────────────────────────────────────────────────────┤
│  TRANSCRIPT DIFF                                             │
│  Gold:   "I want to book a comprehensive heart checkup"      │
│  Actual: "I want to book a [compassion] heart checkup"       │
│  1 substitution  |  Word: "comprehensive" -> "compassion"    │
├──────────────────────────────────────────────────────────────┤
│  [R] Re-run  [E] Export JSON  [Q] Quit  [?] Help            │
└──────────────────────────────────────────────────────────────┘
```

## 📊 Key Metrics

### WER (Word Error Rate)
Measures transcription accuracy by comparing what the STT model transcribed against a gold standard reference. Computed as `(Substitutions + Deletions + Insertions) / Total Reference Words`. A WER > 5% typically indicates environmental noise interference or model limitations. A WER > 15% indicates the pipeline is producing highly unreliable semantic outputs.

### TTFT (Time to First Token)
The most critical latency metric in conversational AI. This measures the time from the end of user speech to the first audible bot response token. EchoTrace measures this by probing the LLM (Groq) in streaming mode. A TTFT > 1000ms feels "slow" to human users. Above 2000ms, user abandonment rates increase significantly.

### SNR (Signal-to-Noise Ratio)
The ratio of speech energy to background noise energy, measured in decibels (dB). Low SNR (< 20dB) directly degrades downstream STT accuracy. EchoTrace calculates this using RMS energy analysis of VAD-segmented audio.

## 🎧 Generating Dataset Fixtures - TEST Data

EchoTrace ships with a built-in utility to generate realistic benchmark datasets from standard open-source audio datasets (e.g., LibriSpeech). This acts as a "Tier 2" integration test suite.

```bash
echotrace generate-fixtures \
    --speech-dir path/to/librispeech/flac_files \
    --noise-dir path/to/freesound_noise/ \
    --out-dir tests/fixtures/my_flac_dataset \
    --max-clips 50
```

This command will automatically mix clean speech with noise at varying Signal-to-Noise levels (e.g., 20dB, 10dB, 5dB) and generate an `echotrace-fixtures.toml` manifest file compatible with the benchmarking suite.

## 🏗️ Architecture

EchoTrace utilizes a robust asynchronous core leveraging `asyncio.gather` for parallel stage execution, wrapping results into a terminal visualization powered by Textual.

```text
CLI (Typer) -> Orchestrator (asyncio.gather)
                 |-- Task A: Signal Analysis (librosa)
                 |-- Task B: Transcription Audit (faster-whisper tiny vs large-v2)
                 |-- Task C: LLM Probe (Groq TTFT or mock)
                       |
                 Aggregator -> Reliability Score + Warnings
                       |
                 Textual TUI (reactive widgets)
```

Every pipeline module is instrumented with a `@timed` decorator that feeds the internal `TelemetryCollector`, serving as the primary source of truth for the visualization waterfall.

## Configuration

To use the real LLM TTFT active probe, export an valid Groq API key:
```bash
export GROQ_API_KEY="gsk_..."
```
Without this key, EchoTrace will gracefully fall back to a mock simulation (simulating 800 - 2000ms latency) and label the widget as `[MOCK]`.

## 🤝 Contributing

Extending EchoTrace visually and functionally requires minimal plumbing. Adding a new architectural analyzer typically modifies a maximum of three locations:

1. Create a new analyzer module in `echotrace/analyzers/`:
```python
from echotrace.telemetry import TelemetryCollector, timed

class MyAnalyzer:
    def __init__(self, collector: TelemetryCollector):
        self._collector = collector

    @timed("my_analysis")
    async def analyze(self, audio_path: str, **kwargs):
        # Implementation details
        return {"result": "value", "_collector": self._collector}
```
2. Integrate the new analyzer class into `echotrace/orchestrator.py` via `asyncio.gather`.
3. Create a reactive UI representation component in `echotrace/tui/widgets/`.

## License

MIT License
