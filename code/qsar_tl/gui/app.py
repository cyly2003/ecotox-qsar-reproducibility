from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qsar_tl.config import ConfigError, load_config, parse_execution_config
from qsar_tl.gui.job_model import JobStatus, TrainingJob
from qsar_tl.gui.process_runner import (
    PYSIDE6_ERROR_MESSAGE,
    build_training_command,
    create_qt_training_process_runner,
)

try:
    from PySide6.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QStackedWidget,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:  # pragma: no cover - depends on local environment.
    raise RuntimeError(PYSIDE6_ERROR_MESSAGE) from exc

from qsar_tl.gui.widgets import (
    ConfigPreviewWidget,
    LogViewWidget,
    MetricsMonitorWidget,
    NotesEditorWidget,
    OutputPathWidget,
    Page,
    TrainingQueueWidget,
)


DEFAULT_CONFIG_PATH = Path("configs/experiment.example.yaml")


@dataclass
class ConsoleState:
    config_path: Path = DEFAULT_CONFIG_PATH
    config: dict[str, Any] = field(default_factory=dict)
    jobs: dict[str, TrainingJob] = field(default_factory=dict)
    active_notes: str = ""

    def add_job(self, job: TrainingJob) -> None:
        self.jobs[job.job_id] = job


class MainWindow(QMainWindow):
    """Main PySide6 window for the ECOTOX-QSAR training console."""

    def __init__(self) -> None:
        super().__init__()
        self.state = ConsoleState()
        self.runner = create_qt_training_process_runner(self)
        self._connect_runner()

        self.setWindowTitle("ECOTOX-QSAR Training Console")
        self.resize(1180, 760)
        self._build_ui()
        self._apply_style()
        self._load_config(DEFAULT_CONFIG_PATH)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        self.nav = QListWidget()
        self.nav.addItems(["实验配置", "训练队列", "实时日志", "指标监控", "输出管理", "实验备注"])
        self.nav.setMaximumWidth(180)

        self.config_preview = ConfigPreviewWidget()
        self.config_preview.configLoadRequested.connect(self._load_config_from_text)
        self.queue = TrainingQueueWidget()
        self.queue.startRequested.connect(self._start_job)
        self.logs = LogViewWidget()
        self.metrics = MetricsMonitorWidget()
        self.outputs = OutputPathWidget()
        self.notes = NotesEditorWidget()
        self.notes.notesChanged.connect(self._set_active_notes)

        self.pages = QStackedWidget()
        self.pages.addWidget(self._config_page())
        self.pages.addWidget(Page("训练队列", self.queue))
        self.pages.addWidget(Page("实时日志", self.logs))
        self.pages.addWidget(Page("指标监控", self.metrics))
        self.pages.addWidget(Page("输出管理", self.outputs))
        self.pages.addWidget(Page("实验备注", self.notes))

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)
        layout.addWidget(self.nav)
        layout.addWidget(self.pages, stretch=1)
        self.setCentralWidget(root)

    def _config_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        title = QLabel("实验配置")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        layout.addWidget(self.config_preview, stretch=1)

        controls = QHBoxLayout()
        add_remote = QPushButton("加入远程训练队列")
        add_remote.clicked.connect(lambda: self._create_job("remote"))
        add_local = QPushButton("加入本机调试队列")
        add_local.clicked.connect(lambda: self._create_job("local"))
        controls.addStretch()
        controls.addWidget(add_local)
        controls.addWidget(add_remote)
        layout.addLayout(controls)
        return page

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                font-size: 13px;
            }
            QListWidget {
                border: 1px solid #d0d7de;
                background: #f6f8fa;
            }
            QListWidget::item {
                padding: 10px 12px;
            }
            QListWidget::item:selected {
                background: #0969da;
                color: white;
            }
            QLabel#pageTitle {
                font-size: 20px;
                font-weight: 600;
                padding: 4px 0 10px 0;
            }
            QPushButton {
                padding: 6px 12px;
            }
            QPlainTextEdit, QLineEdit, QTableWidget {
                border: 1px solid #d0d7de;
            }
            """
        )

    def _connect_runner(self) -> None:
        self.runner.started.connect(self._mark_started)
        self.runner.output.connect(self._append_job_log)
        self.runner.error.connect(self._append_job_error)
        self.runner.finished.connect(self._mark_finished)

    def _load_config_from_text(self, raw_path: str) -> None:
        path = Path(raw_path.strip() or DEFAULT_CONFIG_PATH)
        self._load_config(path)

    def _load_config(self, path: Path) -> None:
        try:
            config = load_config(path)
        except ConfigError as exc:
            self._show_error("配置载入失败", str(exc))
            return

        self.state.config_path = path
        self.state.config = config
        self.config_preview.set_config_data(path, config)
        self.outputs.set_from_config(path, config)
        self.logs.append_log(f"[config] loaded {path}")

    def _create_job(self, execution_mode: str) -> None:
        if not self.state.config:
            self._show_error("缺少配置", "请先载入实验配置文件。")
            return

        job = TrainingJob(
            config_path=self.state.config_path,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            notes=self.state.active_notes,
        )
        execution = parse_execution_config(self.state.config)
        try:
            command = build_training_command(
                job,
                local_python=execution.local_python,
                remote_host=execution.remote_host,
                remote_port=execution.remote_port,
                remote_user=execution.remote_user,
                remote_project_dir=execution.remote_project_dir,
                remote_python=execution.remote_python,
            )
        except ValueError as exc:
            self._show_error("执行配置不完整", str(exc))
            return
        job.set_command(command)
        self.state.add_job(job)
        self.queue.upsert_job(job)
        self.logs.append_log(f"[queue] {job.job_id} {execution_mode} {' '.join(command)}")
        self.nav.setCurrentRow(1)

    def _start_job(self, job_id: str) -> None:
        job = self.state.jobs.get(job_id)
        if job is None:
            self._show_error("任务不存在", f"未找到任务: {job_id}")
            return
        if job.is_terminal:
            self._show_error("任务已结束", f"任务 {job_id} 当前状态为 {job.status.value}。")
            return
        self.runner.start(job, dry_run=True)
        self.nav.setCurrentRow(2)

    def _mark_started(self, job_id: str) -> None:
        job = self.state.jobs.get(job_id)
        if job is None:
            return
        job.set_status(JobStatus.RUNNING)
        self.queue.upsert_job(job)

    def _append_job_log(self, job_id: str, message: str) -> None:
        self.logs.append_log(f"[{job_id}] {message}")

    def _append_job_error(self, job_id: str, message: str) -> None:
        job = self.state.jobs.get(job_id)
        if job is not None:
            job.set_status(JobStatus.FAILED)
            self.queue.upsert_job(job)
        self.logs.append_log(f"[{job_id}][stderr] {message}")

    def _mark_finished(self, job_id: str, exit_code: int) -> None:
        job = self.state.jobs.get(job_id)
        if job is None:
            return
        job.set_status(JobStatus.SUCCEEDED if exit_code == 0 else JobStatus.FAILED)
        self.queue.upsert_job(job)
        self.logs.append_log(f"[{job_id}] finished with exit code {exit_code}")

    def _set_active_notes(self, notes: str) -> None:
        self.state.active_notes = notes

    def _show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
