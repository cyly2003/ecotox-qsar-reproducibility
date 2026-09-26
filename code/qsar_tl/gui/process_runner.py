from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from qsar_tl.gui.job_model import TrainingJob

PYSIDE6_ERROR_MESSAGE = (
    "PySide6 is required to run the ECOTOX-QSAR desktop GUI. "
    "Install it in the active conda environment with: "
    "conda install -c conda-forge pyside6"
)


def _import_qt_core() -> tuple[Any, Any, Any, Any]:
    try:
        from PySide6.QtCore import QObject, QProcess, Signal, Slot
    except Exception as exc:  # pragma: no cover - depends on local Qt install.
        raise RuntimeError(PYSIDE6_ERROR_MESSAGE) from exc
    return QObject, QProcess, Signal, Slot


def ensure_qt_available() -> None:
    _import_qt_core()


def build_local_training_command(config_path: Path, python_exe: str | None = None) -> list[str]:
    python = python_exe or "python"
    return [
        python,
        "-m",
        "qsar_tl.training.train",
        "--config",
        str(config_path),
    ]


def build_remote_training_command(
    config_path: Path,
    *,
    host: str,
    user: str,
    project_dir: str,
    remote_python: str,
    port: int | None = None,
    remote_config_path: str | None = None,
) -> list[str]:
    remote = f"{user}@{host}"
    remote_config = remote_config_path or f"{project_dir.rstrip('/')}/configs/{config_path.name}"
    remote_command = (
        f"cd {shlex.quote(project_dir)} && "
        f"{shlex.quote(remote_python)} -m qsar_tl.training.train "
        f"--config {shlex.quote(remote_config)}"
    )
    command = ["ssh"]
    if port is not None:
        command.extend(["-p", str(port)])
    command.extend([remote, remote_command])
    return command


def build_training_command(
    job: TrainingJob,
    *,
    local_python: str | None = None,
    remote_host: str | None = None,
    remote_user: str | None = None,
    remote_project_dir: str | None = None,
    remote_python: str | None = None,
    remote_port: int | None = None,
) -> list[str]:
    if job.execution_mode == "local":
        return build_local_training_command(job.config_path, python_exe=local_python)
    missing = [
        name
        for name, value in {
            "execution.remote.host": remote_host,
            "execution.remote.user": remote_user,
            "execution.remote.project_dir": remote_project_dir,
            "execution.remote.python": remote_python,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing remote execution settings: {', '.join(missing)}")
    return build_remote_training_command(
        job.config_path,
        host=remote_host or "",
        user=remote_user or "",
        project_dir=remote_project_dir or "",
        remote_python=remote_python or "",
        port=remote_port,
        remote_config_path=job.remote_config_path,
    )


def create_qt_training_process_runner(parent: object | None = None) -> object:
    QObject, QProcess, Signal, Slot = _import_qt_core()

    class QtTrainingProcessRunner(QObject):
        """Qt-friendly wrapper around QProcess for future local/remote training."""

        started = Signal(str)
        output = Signal(str, str)
        error = Signal(str, str)
        finished = Signal(str, int)

        def __init__(self, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._processes: dict[str, QProcess] = {}

        def start(self, job: TrainingJob, *, dry_run: bool = True) -> None:
            """Start a job or emit a dry-run command without blocking the GUI."""
            command = job.command or build_training_command(job)
            job.set_command(command)
            if dry_run:
                self.started.emit(job.job_id)
                self.output.emit(job.job_id, f"[dry-run] {' '.join(command)}")
                self.finished.emit(job.job_id, 0)
                return

            process = QProcess(self)
            process.setProgram(command[0])
            process.setArguments(command[1:])
            process.readyReadStandardOutput.connect(
                lambda job_id=job.job_id, proc=process: self._emit_stdout(job_id, proc)
            )
            process.readyReadStandardError.connect(
                lambda job_id=job.job_id, proc=process: self._emit_stderr(job_id, proc)
            )
            process.finished.connect(
                lambda exit_code, _status, job_id=job.job_id: self._emit_finished(job_id, exit_code)
            )
            process.errorOccurred.connect(
                lambda _error, job_id=job.job_id, proc=process: self.error.emit(
                    job_id, proc.errorString()
                )
            )
            self._processes[job.job_id] = process
            self.started.emit(job.job_id)
            process.start()

        @Slot(str)
        def cancel(self, job_id: str) -> None:
            process = self._processes.get(job_id)
            if process is not None:
                process.kill()

        def _emit_stdout(self, job_id: str, process: QProcess) -> None:
            text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
            if text:
                self.output.emit(job_id, text.rstrip())

        def _emit_stderr(self, job_id: str, process: QProcess) -> None:
            text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
            if text:
                self.error.emit(job_id, text.rstrip())

        def _emit_finished(self, job_id: str, exit_code: int) -> None:
            self._processes.pop(job_id, None)
            self.finished.emit(job_id, exit_code)

    return QtTrainingProcessRunner(parent)
