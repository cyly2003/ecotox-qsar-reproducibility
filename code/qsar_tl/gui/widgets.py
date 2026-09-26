from __future__ import annotations

from pathlib import Path
from typing import Any

from qsar_tl.gui.job_model import JobStatus, TrainingJob
from qsar_tl.gui.process_runner import PYSIDE6_ERROR_MESSAGE

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - depends on local environment.
    raise RuntimeError(PYSIDE6_ERROR_MESSAGE) from exc


def _format_dt(value: object) -> str:
    if value is None:
        return "-"
    return str(value).replace("+00:00", " UTC")


class ConfigPreviewWidget(QWidget):
    configLoadRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("configs/experiment.example.yaml")
        self.load_button = QPushButton("载入配置")
        self.load_button.clicked.connect(lambda: self.configLoadRequested.emit(self.path_edit.text()))
        controls.addWidget(QLabel("配置文件"))
        controls.addWidget(self.path_edit, stretch=1)
        controls.addWidget(self.load_button)
        layout.addLayout(controls)

        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("载入 YAML/JSON 后将在此预览关键配置。")
        layout.addWidget(self.preview, stretch=1)

    def set_config_path(self, path: Path) -> None:
        self.path_edit.setText(str(path))

    def config_path(self) -> Path:
        return Path(self.path_edit.text().strip())

    def set_config_data(self, path: Path, config: dict[str, Any]) -> None:
        self.set_config_path(path)
        lines = [
            f"文件: {path}",
            f"项目: {config.get('project', {}).get('name', '-')}",
            f"执行模式: {config.get('execution', {}).get('mode', '-')}",
            f"输出目录: {config.get('paths', {}).get('output_dir', '-')}",
            f"日志目录: {config.get('paths', {}).get('logs_dir', '-')}",
            "",
            "关键训练参数",
        ]
        training = config.get("training", {})
        for key in ("batch_size", "epochs", "learning_rate"):
            lines.append(f"- {key}: {training.get(key, '-')}")
        metrics = config.get("evaluation", {}).get("metrics", [])
        lines.extend(["", f"评价指标: {', '.join(metrics) if metrics else '-'}"])
        self.preview.setPlainText("\n".join(lines))


class TrainingQueueWidget(QWidget):
    startRequested = Signal(str)

    HEADERS = ["任务 ID", "位置", "配置", "状态", "创建时间", "更新时间", "命令"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)

        self.table = QTableWidget(0, len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, stretch=1)

        controls = QHBoxLayout()
        self.start_button = QPushButton("启动选中任务")
        self.start_button.clicked.connect(self._emit_selected)
        controls.addStretch()
        controls.addWidget(self.start_button)
        layout.addLayout(controls)

    def upsert_job(self, job: TrainingJob) -> None:
        row = self._row_for_job(job.job_id)
        if row is None:
            row = self.table.rowCount()
            self.table.insertRow(row)
        values = [
            job.job_id,
            job.execution_mode,
            str(job.config_path),
            job.status.value,
            _format_dt(job.created_at),
            _format_dt(job.updated_at),
            job.command_text or "-",
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 3:
                item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()

    def selected_job_id(self) -> str | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        item = self.table.item(selected[0].row(), 0)
        return item.text() if item is not None else None

    def _row_for_job(self, job_id: str) -> int | None:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == job_id:
                return row
        return None

    def _emit_selected(self) -> None:
        job_id = self.selected_job_id()
        if job_id:
            self.startRequested.emit(job_id)


class LogViewWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.clear_button = QPushButton("清空日志")
        self.clear_button.clicked.connect(self.clear)
        controls.addStretch()
        controls.addWidget(self.clear_button)
        layout.addLayout(controls)

        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setPlaceholderText("训练命令、stdout/stderr 和远程同步状态将在此显示。")
        layout.addWidget(self.text, stretch=1)

    def append_log(self, message: str) -> None:
        self.text.appendPlainText(message)

    def clear(self) -> None:
        self.text.clear()


class MetricsMonitorWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("指标监控"))
        self.table = QTableWidget(4, 3)
        self.table.setHorizontalHeaderLabels(["指标", "当前值", "说明"])
        rows = [
            ("R2", "-", "回归解释度"),
            ("RMSE", "-", "均方根误差"),
            ("MAE", "-", "平均绝对误差"),
            ("Huber loss", "-", "鲁棒训练损失"),
        ]
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        layout.addStretch()


class OutputPathWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        group = QGroupBox("输出路径")
        form = QFormLayout(group)
        self.output_dir = QLineEdit()
        self.logs_dir = QLineEdit()
        self.config_file = QLineEdit()
        for field in (self.output_dir, self.logs_dir, self.config_file):
            field.setReadOnly(True)
        form.addRow("模型/图表目录", self.output_dir)
        form.addRow("日志目录", self.logs_dir)
        form.addRow("配置文件", self.config_file)
        layout.addWidget(group)
        layout.addStretch()

    def set_from_config(self, path: Path, config: dict[str, Any]) -> None:
        paths = config.get("paths", {})
        self.output_dir.setText(str(paths.get("output_dir", "-")))
        self.logs_dir.setText(str(paths.get("logs_dir", "-")))
        self.config_file.setText(str(path))


class NotesEditorWidget(QWidget):
    notesChanged = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("实验备注"))
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText("记录训练目的、参数变更、异常现象、毒理学解释和后续验证计划。")
        self.editor.textChanged.connect(lambda: self.notesChanged.emit(self.editor.toPlainText()))
        layout.addWidget(self.editor, stretch=1)

    def notes(self) -> str:
        return self.editor.toPlainText()

    def set_notes(self, text: str) -> None:
        self.editor.setPlainText(text)


class Page(QWidget):
    def __init__(self, title: str, body: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        layout.addWidget(title_label)
        layout.addWidget(body, stretch=1)

