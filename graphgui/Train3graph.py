#!/usr/bin/env python3
"""Display Train 3 x_g and y_g measurements from PostgreSQL in two live plots.

Put this script beside ``db_config.json`` and run:

    python train_acceleration_viewer.py

Dependencies:

    python -m pip install PySide6 pyqtgraph psycopg2-binary numpy

The window is deliberately limited to two plot widgets.  It is shown before
the database work begins; loading/progress/error text is drawn inside the
plots instead of adding controls around them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psycopg2
from psycopg2 import sql
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget


MAX_VISIBLE_POINTS = 100_000
FETCH_SIZE = 50_000
PANEL_BLUE = "#176783"
VALID_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def identifier(name: str, label: str) -> sql.Identifier:
    """Return a safely quoted simple SQL identifier from trusted config."""
    if not isinstance(name, str) or not name or any(char not in VALID_IDENTIFIER for char in name):
        raise ValueError(f"{label} must contain only letters, digits, and underscores.")
    return sql.Identifier(name)


@dataclass(frozen=True)
class Settings:
    connection: dict[str, Any]
    schema: str
    table: str
    train_column: str
    train_value: Any
    x_column: str
    y_column: str
    order_column: str


def read_settings(config_path: Path) -> Settings:
    """Read connection and column settings, keeping safe useful defaults."""
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Cannot find {config_path.name}. Put it in the same folder as this script "
            "or use --config /path/to/file.json."
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{config_path.name} is not valid JSON: {exc.msg}.") from exc

    if not isinstance(raw, dict):
        raise ValueError("The JSON root must be an object.")

    # Both a nested {"connection": {...}} and a direct credential object work.
    # Whitelisting prevents the optional data/table keys from being sent to psycopg2.
    source_connection = raw.get("connection", raw)
    if not isinstance(source_connection, dict):
        raise ValueError("connection must be an object when supplied.")
    allowed_connection_keys = {
        "host", "port", "user", "password", "connect_timeout", "application_name",
        "sslmode", "sslrootcert", "sslcert", "sslkey", "options",
    }
    connection = {key: value for key, value in source_connection.items() if key in allowed_connection_keys}
    connection["dbname"] = source_connection.get("dbname", source_connection.get("database", "ofc_daqhat_test"))
    for required in ("host", "user", "password"):
        if not connection.get(required):
            raise ValueError(f"Missing connection.{required} in {config_path.name}.")
    connection["port"] = int(connection.get("port", 5432))
    connection.setdefault("connect_timeout", 10)
    connection.setdefault("application_name", "train_acceleration_viewer")

    data = raw.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("data must be an object when supplied.")
    return Settings(
        connection=connection,
        schema=data.get("schema", raw.get("schema", "public")),
        table=data.get("table", raw.get("table", "test_adxl")),
        train_column=data.get("train_column", "train_no"),
        train_value=data.get("train_value", 3),
        x_column=data.get("x_column", "x_g"),
        y_column=data.get("y_column", "y_g"),
        # This should be a monotonically increasing measurement-time/sequence column.
        order_column=data.get("order_column", "id"),
    )


class MinMaxDecimator:
    """Streaming min/max envelope, preserving peaks without retaining all rows."""

    def __init__(self, total_rows: int, max_points: int) -> None:
        self.total_rows = total_rows
        # Two extrema per bucket are emitted, so this limits each plot to max_points.
        self.bucket_count = max(1, min(math.ceil(max_points / 2), total_rows))
        self.index = 0
        self.current_bucket = -1
        self.entries: list[tuple[int, float, float, float, float]] = []
        self._reset_bucket()

    def _reset_bucket(self) -> None:
        self.x_min = (float("inf"), -1.0)
        self.x_max = (float("-inf"), -1.0)
        self.y_min = (float("inf"), -1.0)
        self.y_max = (float("-inf"), -1.0)

    def _flush(self) -> None:
        if self.current_bucket < 0:
            return
        # Store (sample index, x_g, y_g). Each series gets its own extrema.
        self.entries.append((int(self.x_min[1]), self.x_min[0], np.nan, np.nan, np.nan))
        self.entries.append((int(self.x_max[1]), self.x_max[0], np.nan, np.nan, np.nan))
        self.entries.append((int(self.y_min[1]), np.nan, np.nan, self.y_min[0], np.nan))
        self.entries.append((int(self.y_max[1]), np.nan, np.nan, self.y_max[0], np.nan))

    def add(self, x_value: Any, y_value: Any) -> None:
        bucket = min(self.bucket_count - 1, self.index * self.bucket_count // self.total_rows)
        if bucket != self.current_bucket:
            self._flush()
            self.current_bucket = bucket
            self._reset_bucket()

        position = float(self.index)
        try:
            x_number = float(x_value)
            if math.isfinite(x_number):
                if x_number < self.x_min[0]:
                    self.x_min = (x_number, position)
                if x_number > self.x_max[0]:
                    self.x_max = (x_number, position)
        except (TypeError, ValueError):
            pass
        try:
            y_number = float(y_value)
            if math.isfinite(y_number):
                if y_number < self.y_min[0]:
                    self.y_min = (y_number, position)
                if y_number > self.y_max[0]:
                    self.y_max = (y_number, position)
        except (TypeError, ValueError):
            pass
        self.index += 1

    def finish(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self._flush()
        # The compact entries are split by stream and sorted so pyqtgraph draws lines correctly.
        x_points = [(p, x) for p, x, _, _, _ in self.entries if math.isfinite(x)]
        y_points = [(p, y) for p, _, _, y, _ in self.entries if math.isfinite(y)]
        x_points.sort()
        y_points.sort()
        return (
            np.fromiter((p for p, _ in x_points), dtype=np.float64),
            np.fromiter((v for _, v in x_points), dtype=np.float64),
            np.fromiter((p for p, _ in y_points), dtype=np.float64),
            np.fromiter((v for _, v in y_points), dtype=np.float64),
        )


class DatabaseLoader(QObject):
    progress = Signal(int, int)
    loaded = Signal(object, object, object, object, int)
    failed = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

    def load(self) -> None:
        try:
            s = self.settings
            fields = {
                "schema": identifier(s.schema, "schema"),
                "table": identifier(s.table, "table"),
                "train": identifier(s.train_column, "train_column"),
                "x": identifier(s.x_column, "x_column"),
                "y": identifier(s.y_column, "y_column"),
                "order": identifier(s.order_column, "order_column"),
            }
            relation = sql.SQL("{schema}.{table}").format(**fields)
            with psycopg2.connect(**s.connection) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(
                        sql.SQL("SELECT count(*) FROM {relation} WHERE {train} = %s").format(
                            relation=relation, **fields
                        ),
                        (s.train_value,),
                    )
                    total = cursor.fetchone()[0]
                if total == 0:
                    raise RuntimeError(f"No rows found where {s.train_column} = {s.train_value!r}.")

                # A server-side cursor streams the result; it does not allocate all rows in RAM.
                with connection.cursor(name="train_measurement_stream") as cursor:
                    cursor.itersize = FETCH_SIZE
                    cursor.execute(
                        sql.SQL(
                            "SELECT {x}, {y} FROM {relation} WHERE {train} = %s ORDER BY {order}"
                        ).format(relation=relation, **fields),
                        (s.train_value,),
                    )
                    decimator = MinMaxDecimator(total, MAX_VISIBLE_POINTS)
                    for row_number, (x_value, y_value) in enumerate(cursor, start=1):
                        decimator.add(x_value, y_value)
                        if row_number % FETCH_SIZE == 0:
                            self.progress.emit(row_number, total)

            x_index, x_data, y_index, y_data = decimator.finish()
            self.loaded.emit(x_index, x_data, y_index, y_data, total)
        except Exception as exc:  # Shown in-window so configuration errors are actionable.
            self.failed.emit(str(exc))


class TrainGraphWindow(QMainWindow):
    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.setWindowTitle("Train 3 Acceleration")
        self.resize(1400, 760)
        self.setStyleSheet("background: #f5f5f5;")

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(34, 34, 34, 34)
        layout.setSpacing(28)
        self.setCentralWidget(root)

        self.x_plot, self.x_message = self._make_plot("X_g")
        self.y_plot, self.y_message = self._make_plot("Y_g")
        layout.addWidget(self.x_plot, 1)
        layout.addWidget(self.y_plot, 1)

        self.thread = QThread(self)
        self.loader = DatabaseLoader(settings)
        self.loader.moveToThread(self.thread)
        self.thread.started.connect(self.loader.load)
        self.loader.progress.connect(self._set_progress)
        self.loader.loaded.connect(self._render)
        self.loader.failed.connect(self._show_error)
        self.loader.loaded.connect(self.thread.quit)
        self.loader.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.loader.deleteLater)
        self.thread.start()

    @staticmethod
    def _make_plot(title: str) -> tuple[pg.PlotWidget, pg.TextItem]:
        plot = pg.PlotWidget(background=PANEL_BLUE)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=True, y=True)
        plot.showGrid(x=True, y=True, alpha=0.16)
        plot.getPlotItem().hideAxis("left")
        plot.getPlotItem().hideAxis("bottom")
        plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        plot.setTitle(title, color="white", size="15pt")
        message = pg.TextItem("Loading Train 3…", color="white", anchor=(0.5, 0.5))
        plot.addItem(message)
        message.setPos(0.5, 0.5)
        return plot, message

    def _set_progress(self, current: int, total: int) -> None:
        percent = current * 100 / total
        text = f"Loading Train 3… {percent:.0f}%"
        self.x_message.setText(text)
        self.y_message.setText(text)

    def _render(
        self, x_index: np.ndarray, x_data: np.ndarray, y_index: np.ndarray, y_data: np.ndarray, total: int
    ) -> None:
        pen = pg.mkPen("#ffffff", width=1)
        self.x_plot.plot(x_index, x_data, pen=pen, connect="finite", antialias=False)
        self.y_plot.plot(y_index, y_data, pen=pen, connect="finite", antialias=False)
        self.x_plot.removeItem(self.x_message)
        self.y_plot.removeItem(self.y_message)
        self.setWindowTitle(f"Train 3 Acceleration — {total:,} samples")

    def _show_error(self, message: str) -> None:
        text = f"Could not load Train 3\n{message}"
        self.x_message.setText(text)
        self.y_message.setText(text)

    def closeEvent(self, event) -> None:  # Avoid leaving a worker thread alive on close.
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(1500)
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description="Show train x_g/y_g from PostgreSQL.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("db_config.json"),
        help="JSON connection/configuration file (default: db_config.json beside this script)",
    )
    args = parser.parse_args()
    app = QApplication(sys.argv)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
    try:
        settings = read_settings(args.config)
    except Exception as exc:
        # Create the same visual window even for a local configuration problem.
        settings = None
        error = str(exc)
    if settings is None:
        window = QMainWindow()
        window.setWindowTitle("Train 3 Acceleration")
        panel = pg.PlotWidget(background=PANEL_BLUE)
        panel.addItem(pg.TextItem(f"Configuration error\n{error}", color="white", anchor=(0.5, 0.5)))
        window.setCentralWidget(panel)
        window.resize(900, 500)
    else:
        window = TrainGraphWindow(settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
