# Contributing to EchoTrace

Thank you for your interest in contributing to EchoTrace! This guide will help you get started.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/priyavratuniyal/echotrace-cli.git
cd echotrace-cli

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install in development mode with all dev dependencies
make install
# or manually:
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

### Running Tests

```bash
make test
# or
pytest tests/ -v --tb=short
```

### Code Quality

```bash
# Lint check
make lint

# Auto-format
make format

# Type check
make typecheck
```

### Pre-commit Hooks

Pre-commit hooks run automatically on every `git commit`. They will:
- Check and fix formatting with `ruff format`
- Check linting with `ruff check`
- Verify YAML/TOML files
- Block large files and private keys

If a hook fails, fix the issue and re-commit.

## Adding a New Analyzer

EchoTrace is designed to be easily extensible. To add a new pipeline analyzer:

1. Create a new module in `echotrace/analyzers/`:
```python
from echotrace.core.telemetry import TelemetryCollector, timed

class MyAnalyzer:
    def __init__(self, collector: TelemetryCollector):
        self._collector = collector

    @timed("my_analysis")
    async def analyze(self, audio_path: str, **kwargs):
        # Your analysis logic here
        return {"result": "value"}
```

2. Integrate it into `echotrace/orchestrator.py` via `asyncio.gather`
3. Create a reactive UI widget in `echotrace/tui/widgets/`

## Pull Request Checklist

- [ ] Tests added for new behavior
- [ ] All existing tests pass (`pytest tests/`)
- [ ] `ruff check .` passes
- [ ] `ruff format --check .` passes
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Version **NOT** bumped in `pyproject.toml` (that happens on release)

## Release Process

Releases are automated via GitHub Actions. The process:

1. Update `CHANGELOG.md` — move items from `[Unreleased]` to a new version section
2. Bump version in `pyproject.toml`
3. Commit and push to `main`
4. Wait for CI to pass
5. Tag: `git tag v0.X.0 && git push origin v0.X.0`
6. Approve the PyPI deployment when prompted in GitHub

## Code Style

- Line length: 88 characters
- Formatter: `ruff format`
- Linter: `ruff check`
- Type checker: `mypy`
- Quote style: double quotes
