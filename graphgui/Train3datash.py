#!/usr/bin/env python3
"""Fast two-panel Train 3 X_g / Y_g viewer using Datashader.

Every matching row from PostgreSQL is streamed and contributes to the final
Datashader raster. No row sampling and no max-points cutoff are applied.
Datashader rasterizes the full polyline to display resolution, so the result is
an image of the complete signal rather than 28+ million individually selectable
plot vertices.

Place this file next to db_config.json.

Install:
    python -m pip install PySide6 pyqtgraph psycopg2-binary pandas datashader

Config is compatible with the direct JSON style used by the earlier scripts:
{
  "host": "...", "port": 5432, "dbname": "ofc_daqhat_test",
  "user": "...", "password": "...",
  "schema": "public", "table": "test_adxl", "train_no": 3,
  "order_by": null
}

Set order_by to a real sequence/timestamp column (for example "timestamp" or
"id") for chronological/deterministic order. With order_by=null, PostgreSQL
may return rows in an unspecified order, although every returned row is used.
"""
from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Qt, QRectF, Signal
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget

CONFIG_PATH = Path(__file__).with_name("db_config.json")
FETCH_SIZE = 100_000
RASTER_WIDTH = 1500
RASTER_HEIGHT = 650
PANEL_BLUE = "#176783"
LINE_COLOUR = "#FFFFFF"
VALID_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def checked_identifier(value: Any, label: str) -> sql.Identifier:
    if not isinstance(value, str) or not value or any(ch not in VALID_IDENTIFIER for ch in value):
        raise ValueError(f"{label} must contain only letters, digits, and underscores.")
    return sql.Identifier(value)


@dataclass(frozen=True)
class Settings:
    connection: dict[str, Any]
    schema: str
    table: str
    train_column: str
    train_value: Any
    x_column: str
    y_column: str
    order_column: Optional[str]


def read_settings(path: Path = CONFIG_PATH) -> Settings:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path.name}; put it beside this script.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is invalid JSON: {exc.msg}.") from exc
    if not isinstance(raw, dict):
        raise ValueError("The JSON configuration must be an object.")

    # Accept either direct connection settings or {"connection": {...}}.
    connection_source = raw.get("connection", raw)
    if not isinstance(connection_source, dict):
        raise ValueError("The 'connection' setting must be a JSON object.")
    allowed = {
        "host", "port", "dbname", "database", "user", "password",
        "connect_timeout", "application_name", "sslmode", "sslrootcert",
        "sslcert", "sslkey", "options", "keepalives", "keepalives_idle",
        "keepalives_interval", "keepalives_count",
    }
    conn = {k: v for k, v in connection_source.items() if k in allowed}
    conn["dbname"] = connection_source.get(
        "dbname", connection_source.get("database", "ofc_daqhat_test")
    )
    for key in ("host", "dbname", "user", "password"):
        if not conn.get(key):
            raise ValueError(f"Fill in '{key}' in {path.name}.")
    conn["port"] = int(conn.get("port", 5432))
    conn.setdefault("connect_timeout", 15)
    conn.setdefault("application_name", "train3_datashader_viewer")
    # TCP keepalives help detect a dead network path; they do not override any
    # provider-side query or connection limits.
    conn.setdefault("keepalives", 1)
    conn.setdefault("keepalives_idle", 30)
    conn.setdefault("keepalives_interval", 10)
    conn.setdefault("keepalives_count", 6)

    data = raw.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("'data' must be a JSON object when supplied.")
    schema = data.get("schema", raw.get("schema", "public"))
    table = data.get("table", raw.get("table", "test_adxl"))
    train_column = data.get("train_column", raw.get("train_column", "train_no"))
    train_value = data.get("train_value", raw.get("train_no", 3))
    x_column = data.get("x_column", raw.get("x_column", "x_g"))
    y_column = data.get("y_column", raw.get("y_column", "y_g"))

    # Support both new data.order_column and older root order_by settings.
    if "order_column" in data:
        order_column = data["order_column"]
    elif "order_by" in raw:
        order_column = raw["order_by"]
    else:
        order_column = "timestamp"
    if order_column is not None and (not isinstance(order_column, str) or not order_column):
        raise ValueError("order_by/order_column must be a column name or null.")

    # Validate names before starting any SQL work.
    for name, label in ((schema, "schema"), (table, "table"), (train_column, "train_column"),
                        (x_column, "x_column"), (y_column, "y_column")):
        checked_identifier(name, label)
    if order_column is not None:
        checked_identifier(order_column, "order_by/order_column")

    return Settings(conn, schema, table, train_column, train_value, x_column, y_column, order_column)


def finite_range(low: Any, high: Any, fallback: tuple[float, float] = (-1.0, 1.0)) -> tuple[float, float]:
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError, OverflowError):
        return fallback
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return fallback
    if lo == hi:
        pad = max(abs(lo) * 0.01, 1e-6)
        return lo - pad, hi + pad
    # A tiny margin prevents extrema from landing exactly on a canvas edge.
    pad = (hi - lo) * 0.005
    return lo - pad, hi + pad


def rows_to_float_array(rows: list[tuple[Any, Any]], col: int) -> np.ndarray:
    result = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        value = row[col]
        if value is None:
            result[i] = np.nan
        else:
            try:
                number = float(value)
                result[i] = number if math.isfinite(number) else np.nan
            except (TypeError, ValueError, OverflowError):
                result[i] = np.nan
    return result


class DatabaseRasterWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object, tuple, tuple, int, float)
    failed = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

    def run(self) -> None:
        started = time.perf_counter()
        try:
            s = self.settings
            ids = {
                "schema": checked_identifier(s.schema, "schema"),
                "table": checked_identifier(s.table, "table"),
                "train": checked_identifier(s.train_column, "train_column"),
                "x": checked_identifier(s.x_column, "x_column"),
                "y": checked_identifier(s.y_column, "y_column"),
            }
            relation = sql.SQL("{}.{}").format(ids["schema"], ids["table"])

            self.progress.emit("Connecting to PostgreSQL…")
            with psycopg2.connect(**s.connection) as connection:
                # One compact result gives an exact row count and y bounds. This
                # avoids a second 28.8M-row transfer just to determine plot ranges.
                self.progress.emit("Getting row count and graph ranges…")
                stats_query = sql.SQL(
                    "SELECT COUNT(*), MIN({x}), MAX({x}), MIN({y}), MAX({y}) "
                    "FROM {relation} WHERE {train} = %s"
                ).format(relation=relation, **ids)
                with connection.cursor() as cur:
                    cur.execute(stats_query, (s.train_value,))
                    total, x_min, x_max, y_min, y_max = cur.fetchone()

                total = int(total or 0)
                if total == 0:
                    raise RuntimeError(f"No rows found where {s.train_column} = {s.train_value!r}.")
                x_range = finite_range(x_min, x_max)
                y_range = finite_range(y_min, y_max)
                sample_range = (0.5, float(total) + 0.5)

                # Separate fixed-size raster accumulators; the row count is
                # bounded by display pixels, NOT by a sampled point limit.
                canvas_x = ds.Canvas(
                    plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT,
                    x_range=sample_range, y_range=x_range,
                )
                canvas_y = ds.Canvas(
                    plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT,
                    x_range=sample_range, y_range=y_range,
                )
                accum_x = np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=np.uint32)
                accum_y = np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=np.uint32)

                query = sql.SQL("SELECT {x}, {y} FROM {relation} WHERE {train} = %s").format(
                    relation=relation, **ids
                )
                if s.order_column:
                    query += sql.SQL(" ORDER BY {} ASC NULLS LAST").format(
                        checked_identifier(s.order_column, "order_by/order_column")
                    )

                # A named/server-side cursor plus fetchmany avoids materializing
                # millions of Python tuples in RAM. Every row is passed to both
                # Datashader canvases; only the final pixel rasters are retained.
                self.progress.emit(f"Rasterizing all {total:,} Train 3 rows… 0%")
                template_x = None
                template_y = None
                with connection.cursor(name="train3_datashader_stream") as cur:
                    cur.itersize = FETCH_SIZE
                    cur.execute(query, (s.train_value,))
                    row_number = 0
                    previous: Optional[tuple[float, float, float]] = None
                    while True:
                        rows = cur.fetchmany(FETCH_SIZE)
                        if not rows:
                            break
                        count = len(rows)
                        x_values = rows_to_float_array(rows, 0)
                        y_values = rows_to_float_array(rows, 1)
                        samples = np.arange(row_number + 1, row_number + count + 1, dtype=np.float64)

                        # Carry the previous batch's final vertex into this batch
                        # so the line segment between batches is also rasterized.
                        if previous is not None:
                            samples = np.concatenate(([previous[0]], samples))
                            x_values = np.concatenate(([previous[1]], x_values))
                            y_values = np.concatenate(([previous[2]], y_values))

                        frame = pd.DataFrame({"sample": samples, "x_g": x_values, "y_g": y_values})
                        ax = canvas_x.line(frame, x="sample", y="x_g", agg=ds.count())
                        ay = canvas_y.line(frame, x="sample", y="y_g", agg=ds.count())
                        template_x, template_y = ax, ay
                        # Counts can be added because every chunk uses identical
                        # ranges and dimensions. uint32 safely covers this dataset's
                        # maximum possible count (under 4.3 billion total rows).
                        np.add(accum_x, np.asarray(ax.values, dtype=np.uint32), out=accum_x)
                        np.add(accum_y, np.asarray(ay.values, dtype=np.uint32), out=accum_y)

                        row_number += count
                        previous = (float(row_number), float(x_values[-1]), float(y_values[-1]))
                        if row_number % (FETCH_SIZE * 2) == 0 or row_number == total:
                            pct = min(100.0, row_number * 100.0 / total)
                            self.progress.emit(f"Rasterizing all {total:,} Train 3 rows… {pct:.1f}%")

            if row_number != total:
                raise RuntimeError(
                    f"The matching row count changed during loading: expected {total:,}, "
                    f"but streamed {row_number:,}. Please retry against a stable dataset."
                )
            if template_x is None or template_y is None:
                raise RuntimeError("No rows were returned by the streaming query.")

            # Shade only the small fixed pixel arrays. Empty pixels remain
            # transparent so the blue PlotWidget background shows through.
            self.progress.emit("Finishing X_g and Y_g images…")
            if np.any(accum_x):
                x_template = template_x.copy(data=accum_x)
                x_rgba = np.asarray(
                    tf.shade(x_template, cmap=[LINE_COLOUR], how="eq_hist", alpha=255).to_pil()
                )
            else:
                x_rgba = np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)
            if np.any(accum_y):
                y_template = template_y.copy(data=accum_y)
                y_rgba = np.asarray(
                    tf.shade(y_template, cmap=[LINE_COLOUR], how="eq_hist", alpha=255).to_pil()
                )
            else:
                y_rgba = np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)
            self.completed.emit(
                np.ascontiguousarray(np.flipud(x_rgba)),
                np.ascontiguousarray(np.flipud(y_rgba)),
                x_range, y_range, total, time.perf_counter() - started,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class Train3Window(QMainWindow):
    def __init__(self, settings: Optional[Settings], config_error: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Train 3 — X_g and Y_g")
        self.resize(1400, 760)
        self.setStyleSheet("QMainWindow { background: #f4f5f6; }")
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        self.setCentralWidget(root)
        self.x_plot, self.x_message, self.x_image = self._make_plot("X_g")
        self.y_plot, self.y_message, self.y_image = self._make_plot("Y_g")
        layout.addWidget(self.x_plot, 1)
        layout.addWidget(self.y_plot, 1)

        self.thread: Optional[QThread] = None
        self.worker: Optional[DatabaseRasterWorker] = None
        if config_error:
            self._show_error(config_error)
        elif settings is not None:
            self.thread = QThread(self)
            self.worker = DatabaseRasterWorker(settings)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(self._set_message)
            self.worker.completed.connect(self._show_images)
            self.worker.failed.connect(self._show_error)
            self.worker.completed.connect(self.thread.quit)
            self.worker.failed.connect(self.thread.quit)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.start()

    @staticmethod
    def _make_plot(title: str):
        plot = pg.PlotWidget(background=PANEL_BLUE)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=True, y=True)
        plot.showGrid(x=True, y=True, alpha=0.12)
        plot.setTitle(title, color="white", size="15pt")
        plot.getPlotItem().hideAxis("left")
        plot.getPlotItem().hideAxis("bottom")
        plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        message = pg.TextItem("Loading…", color="white", anchor=(0.5, 0.5))
        plot.addItem(message)
        message.setPos(0.5, 0.5)
        # Global imageAxisOrder is set before the window is created for compatibility
        # with both recent and Python-3.9-compatible pyqtgraph releases.
        image = pg.ImageItem()
        plot.addItem(image)
        return plot, message, image

    def _set_message(self, text: str) -> None:
        self.x_message.setText(text)
        self.y_message.setText(text)
        # Keep the status near the middle while the graph has not been drawn.
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)

    def _show_images(self, x_rgba: np.ndarray, y_rgba: np.ndarray,
                     x_range: tuple[float, float], y_range: tuple[float, float],
                     total: int, elapsed: float) -> None:
        if x_rgba.size:
            self.x_image.setImage(x_rgba, autoLevels=False, levels=None)
            self.x_image.setRect(QRectF(0.5, x_range[0], float(total), x_range[1] - x_range[0]))
        if y_rgba.size:
            self.y_image.setImage(y_rgba, autoLevels=False, levels=None)
            self.y_image.setRect(QRectF(0.5, y_range[0], float(total), y_range[1] - y_range[0]))
        self.x_plot.setXRange(0.5, float(total) + 0.5, padding=0)
        self.y_plot.setXRange(0.5, float(total) + 0.5, padding=0)
        self.x_plot.setYRange(x_range[0], x_range[1], padding=0)
        self.y_plot.setYRange(y_range[0], y_range[1], padding=0)
        self.x_plot.removeItem(self.x_message)
        self.y_plot.removeItem(self.y_message)
        self.setWindowTitle(f"Train 3 — X_g and Y_g · {total:,} rows · Datashader")
        print(f"Rendered all {total:,} Train 3 rows into both Datashader graphs in {elapsed:.1f}s.")

    def _show_error(self, message: str) -> None:
        msg = "Could not load Train 3 data\n\n" + message
        self.x_message.setText(msg)
        self.y_message.setText(msg)
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)
        print(msg, file=sys.stderr)

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait(2000)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    try:
        settings = read_settings()
        error = None
    except Exception as exc:
        settings = None
        error = f"Configuration error: {exc}"
    window = Train3Window(settings, error)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
