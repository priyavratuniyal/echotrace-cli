"""
EchoTrace configuration loader.

Reads settings from (in order of precedence):
1. .echotrace.toml in the current working directory
2. ~/.config/echotrace/config.toml
3. Environment variables
4. Hardcoded defaults
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

USER_CONFIG_DIR = Path.home() / ".config" / "echotrace"
USER_CONFIG_PATH = USER_CONFIG_DIR / "config.toml"
STATE_PATH = USER_CONFIG_DIR / "state.json"

def is_first_run() -> bool:
    """Determine if EchoTrace is running for the very first time."""
    state = load_state()
    return not state.get("first_run_complete", False)

def mark_setup_complete() -> None:
    """Mark that the user has gone through the first run."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.touch()
    state = load_state()
    state["setup_complete"] = True
    state["first_run_complete"] = True
    save_state(state)

def load_state() -> dict:
    """Load lightweight usage state from state.json."""
    import json
    if not STATE_PATH.exists():
        return {
            "first_run_complete": False,
            "demo_run_count": 0,
            "total_analyses": 0,
            "last_analysis": None,
            "setup_complete": False,
        }
    try:
        return json.loads(STATE_PATH.read_text())
    except Exception:
        return {"first_run_complete": False}

def save_state(state: dict) -> None:
    """Persist usage state to state.json."""
    import json
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))

def load_config() -> dict:
    """Load the raw config dictionary."""
    if not USER_CONFIG_PATH.exists():
        return {}
    
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
            
    with open(USER_CONFIG_PATH, "rb") as f:
        try:
            return tomllib.load(f)
        except Exception:
            return {}

def save_config(config: dict) -> None:
    """Save the config dictionary to the TOML file."""
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Simple TOML serializer for our basic dict structure
    lines = []
    
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f"[{section}]")
            for k, v in values.items():
                if isinstance(v, str):
                    lines.append(f'{k} = "{v}"')
                else:
                    lines.append(f'{k} = {v}')
            lines.append("")
            
    with open(USER_CONFIG_PATH, "w") as f:
        f.write("\n".join(lines))
@dataclass
class LLMConfig:
    """LLM provider settings."""

    provider: str = "auto"  # auto | groq | ollama | huggingface | mock
    groq_api_key: str = ""
    groq_model: str = "llama3-8b-8192"
    ollama_model: str = "llama3.2:3b"
    ollama_host: str = "http://localhost:11434"
    hf_api_key: str = ""
    hf_model: str = "microsoft/Phi-3-mini-4k-instruct"


@dataclass
class EchoTraceConfig:
    """Root configuration for EchoTrace."""

    llm: LLMConfig = field(default_factory=LLMConfig)

    @staticmethod
    def load() -> EchoTraceConfig:
        """
        Load config from TOML files and environment variables.

        Priority: env vars override TOML, CWD TOML overrides user TOML.
        """
        config = EchoTraceConfig()

        # 1. Try user-level config
        user_path = Path.home() / ".config" / "echotrace" / "config.toml"
        if user_path.exists():
            config._merge_toml(user_path)

        # 2. Try project-level config (overrides user-level)
        project_path = Path.cwd() / ".echotrace.toml"
        if project_path.exists():
            config._merge_toml(project_path)

        # 3. Environment variables (highest priority)
        config._merge_env()

        logger.debug(
            f"Config loaded: provider={config.llm.provider}, "
            f"groq_key={'set' if config.llm.groq_api_key else 'unset'}, "
            f"hf_key={'set' if config.llm.hf_api_key else 'unset'}"
        )

        return config

    def _merge_toml(self, path: Path) -> None:
        """Merge a TOML file into this config."""
        try:
            if sys.version_info >= (3, 11):
                import tomllib
            else:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[no-redef]

            with open(path, "rb") as f:
                data = tomllib.load(f)

            llm = data.get("llm", {})
            for key in (
                "provider",
                "groq_api_key",
                "groq_model",
                "ollama_model",
                "ollama_host",
                "hf_api_key",
                "hf_model",
            ):
                val = llm.get(key)
                if val is not None and val != "":
                    setattr(self.llm, key, val)

            logger.debug(f"Loaded config from {path}")

        except Exception as e:
            logger.warning(f"Failed to load config from {path}: {e}")

    def _merge_env(self) -> None:
        """Override config with environment variables."""
        env_map = {
            "ECHOTRACE_LLM_PROVIDER": "provider",
            "GROQ_API_KEY": "groq_api_key",
            "ECHOTRACE_GROQ_MODEL": "groq_model",
            "ECHOTRACE_OLLAMA_MODEL": "ollama_model",
            "OLLAMA_HOST": "ollama_host",
            "HF_API_KEY": "hf_api_key",
            "ECHOTRACE_HF_MODEL": "hf_model",
        }
        for env_var, attr in env_map.items():
            val = os.getenv(env_var)
            if val is not None and val != "":
                setattr(self.llm, attr, val)
