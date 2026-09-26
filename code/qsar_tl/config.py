from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(RuntimeError):
    """Raised when an experiment configuration cannot be loaded or validated."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is required to load YAML configs.") from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        import json

        data = json.loads(text)
    else:
        raise ConfigError(f"Unsupported config format: {suffix}")

    if not isinstance(data, dict):
        raise ConfigError("Top-level config must be a mapping.")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = config_path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError("PyYAML is required to save YAML configs.") from exc
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    elif suffix == ".json":
        import json

        text = json.dumps(config, ensure_ascii=False, indent=2)
    else:
        raise ConfigError(f"Unsupported config format: {suffix}")

    config_path.write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class ExecutionConfig:
    mode: str
    local_python: str | None = None
    remote_host: str | None = None
    remote_port: int | None = None
    remote_user: str | None = None
    remote_project_dir: str | None = None
    remote_python: str | None = None


def parse_execution_config(config: dict[str, Any]) -> ExecutionConfig:
    execution = config.get("execution", {})
    remote = execution.get("remote", {}) if isinstance(execution.get("remote", {}), dict) else {}
    return ExecutionConfig(
        mode=execution.get("mode", "local"),
        local_python=execution.get("local_python"),
        remote_host=remote.get("host"),
        remote_port=_parse_optional_int(remote.get("port"), "execution.remote.port"),
        remote_user=remote.get("user"),
        remote_project_dir=remote.get("project_dir"),
        remote_python=remote.get("python"),
    )


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be an integer.") from exc
