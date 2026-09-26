from __future__ import annotations

import subprocess
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from qsar_tl.config import parse_execution_config


class Runner(Protocol):
    def run(self, config_path: Path, dry_run: bool = False) -> None:
        ...


@dataclass
class LocalRunner:
    python_exe: str | None = None

    def run(self, config_path: Path, dry_run: bool = False) -> None:
        python = self.python_exe or "python"
        command = [python, "-m", "qsar_tl.training.train", "--config", str(config_path)]
        if dry_run:
            print(format_command(command))
            return
        subprocess.run(command, check=True)


@dataclass
class RemoteRunner:
    host: str
    user: str
    project_dir: str
    remote_python: str
    port: int | None = None

    def run(self, config_path: Path, dry_run: bool = False) -> None:
        remote = f"{self.user}@{self.host}"
        remote_config = f"{self.project_dir}/configs/{config_path.name}"
        remote_command = (
            f"cd {shlex.quote(self.project_dir)} && "
            f"{shlex.quote(self.remote_python)} -m qsar_tl.training.train "
            f"--config {shlex.quote(remote_config)}"
        )
        ssh_command = ["ssh"]
        if self.port is not None:
            ssh_command.extend(["-p", str(self.port)])
        command = [
            *ssh_command,
            remote,
            remote_command,
        ]
        if dry_run:
            print(format_command(command))
            return
        subprocess.run(command, check=True)


def build_runner(config: dict[str, Any]) -> Runner:
    execution = parse_execution_config(config)
    if execution.mode == "remote":
        missing = [
            name
            for name, value in {
                "remote_host": execution.remote_host,
                "remote_user": execution.remote_user,
                "remote_project_dir": execution.remote_project_dir,
                "remote_python": execution.remote_python,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing remote execution settings: {', '.join(missing)}")
        return RemoteRunner(
            host=execution.remote_host or "",
            user=execution.remote_user or "",
            project_dir=execution.remote_project_dir or "",
            remote_python=execution.remote_python or "",
            port=execution.remote_port,
        )
    return LocalRunner(python_exe=execution.local_python)


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)
