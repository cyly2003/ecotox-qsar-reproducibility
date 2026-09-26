from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal
from uuid import uuid4


ExecutionMode = Literal["local", "remote"]


class JobStatus(str, Enum):
    """Lifecycle state for one GUI-managed training job."""

    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TrainingJob:
    """State tracked by the training console for local or remote runs."""

    config_path: Path
    execution_mode: ExecutionMode = "remote"
    remote_config_path: str | None = None
    command: list[str] = field(default_factory=list)
    status: JobStatus = JobStatus.QUEUED
    notes: str = ""
    job_id: str = field(default_factory=lambda: uuid4().hex[:12])
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    def __post_init__(self) -> None:
        self.config_path = Path(self.config_path)
        if self.execution_mode not in {"local", "remote"}:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")
        if not isinstance(self.status, JobStatus):
            self.status = JobStatus(str(self.status))

    @property
    def command_text(self) -> str:
        return " ".join(self.command)

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }

    def set_command(self, command: list[str]) -> None:
        self.command = list(command)
        self.touch()

    def set_notes(self, notes: str) -> None:
        self.notes = notes
        self.touch()

    def set_status(self, status: JobStatus | str) -> None:
        next_status = status if isinstance(status, JobStatus) else JobStatus(status)
        self.status = next_status
        now = utc_now()
        self.updated_at = now
        if next_status == JobStatus.RUNNING and self.started_at is None:
            self.started_at = now
        if next_status in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        }:
            self.finished_at = now

    def touch(self) -> None:
        self.updated_at = utc_now()
