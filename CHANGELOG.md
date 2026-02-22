# Changelog

All notable changes to EchoTrace are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

---

## [Unreleased]

### Added
- GitHub Actions CI/CD pipeline
- PyPI publishing workflow
- Security scanning with pip-audit
- Pre-commit hooks for code quality

---

## [0.1.0] — 2026-02-23

### Added
- Initial release of EchoTrace Voice AI Reliability Profiler
- `echotrace demo` command with bundled sample audio
- `echotrace analyze` with SNR calculation, WER measurement, latency waterfall
- `echotrace setup` interactive configuration wizard
- Groq API integration for TTFT measurement
- Ollama integration for local LLM inference
- Mock LLM provider for offline use
- Rich terminal UI with latency waterfall and transcript diff
- Pydantic validation for all CLI inputs
- JSON report export
- Signal analysis with VAD and RMS energy
- Word-level transcript diff with substitution/insertion/deletion tracking
- `echotrace generate-fixtures` for benchmark dataset creation

### Known Issues
- faster-whisper large-v2 download prompt not yet implemented
- Ollama integration is experimental
