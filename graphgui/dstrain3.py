#!/usr/bin/env python3
"""Fast two-panel Train 3 X_g / Y_g viewer using Datashader (optimized).

Same behaviour as before: every matching row is used, no sampling, same
Datashader rasterization, same window. What changed is HOW the data is moved
and rasterized:

  1. Download: PostgreSQL binary COPY (COPY (SELECT ...) TO STDOUT BINARY)
     instead of a server-side cursor + fetchmany. The binary stream is parsed
     with NumPy in big blocks, so there is no per-row Python tuple, no
     Decimal/float() conversion loop and no text parsing.
  2. One pass over the table: the separate COUNT/MIN/MAX scan is gone. Ranges
     and row count are computed from the downloaded arrays (they fit in RAM:
     28.8M rows is ~230 MB as float32 or ~460 MB as float64, per axis pair).
  3. Rasterization runs on all CPU cores (Datashader's numba kernels release
     the GIL), with large chunks, X and Y rasterized as independent tasks.

Place this file next to db_config.json. Same config format as before.
Run: python dstrain.py
GUI-only check: python dstrain.py --check-gui
Alternate config location: python dstrain.py --config "C:/path/db_config.json"
Closing during loading waits for the current data operation to finish.

Install (nothing new needed):
    python -m pip install PySide6 pyqtgraph psycopg2-binary pandas datashader

Speed tip for the database itself (biggest remaining win if the server sorts):
    CREATE INDEX CONCURRENTLY idx_test_adxl_train_ts
        ON public.test_adxl (train_no, "timestamp");
With that index PostgreSQL can stream the rows already in order instead of
sorting ~28.8M rows before sending the first byte.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import datashader as ds
import datashader.transfer_functions as tf
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
# All widgets and plots must use the same Qt binding.
os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"
import PySide6
from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Qt, QRectF, Signal, Slot
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget
import pyqtgraph as pg

if pg.Qt.QT_LIB != "PySide6":
    raise RuntimeError("pyqtgraph already uses another Qt binding. Restart in a fresh Python process.")

CONFIG_PATH = Path(__file__).resolve().with_name("db_config.json")
RASTER_WIDTH = 1500
RASTER_HEIGHT = 650
PANEL_BLUE = "#176783"
LINE_COLOUR = "#FFFFFF"
VALID_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

# Transfer values as 4-byte floats (~30% less data over the network). g-values
# have far more precision than a 650-pixel-high raster can show. Set to False
# to transfer full 8-byte doubles (identical to the old script's precision).
TRANSFER_FLOAT32 = True
# Rows parsed per NumPy block while downloading, and per Datashader task.
DOWNLOAD_BLOCK_ROWS = 1_000_000
RASTER_WORKERS = min(32, os.cpu_count() or 4)


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
        raise FileNotFoundError(f"Cannot find configuration: {path.resolve()}. Put db_config.json there or use --config PATH.")
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


class BinaryCopyParser:
    """File-like sink for psycopg2 copy_expert(... TO STDOUT WITH (FORMAT BINARY)).

    The query always returns exactly two non-NULL float columns (NULLs are
    turned into NaN in SQL), so every row has the same fixed size and a whole
    block of rows can be decoded with one np.frombuffer call.
    """

    SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
    HEADER_FIXED = 19  # 11 signature + 4 flags + 4 extension length
    TRAILER = b"\xff\xff"

    def __init__(self, float32: bool, block_rows: int,
                 on_block: Optional[Callable[[int], None]] = None) -> None:
        ftype = ">f4" if float32 else ">f8"
        self.field_len = 4 if float32 else 8
        self.native = np.float32 if float32 else np.float64
        self.dtype = np.dtype([
            ("n", ">i2"), ("lx", ">i4"), ("x", ftype), ("ly", ">i4"), ("y", ftype),
        ])
        self.itemsize = self.dtype.itemsize
        self.threshold = block_rows * self.itemsize
        self.on_block = on_block
        self.buf = bytearray()
        self.header_done = False
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.rows = 0
        self.x_min: Optional[float] = None
        self.x_max: Optional[float] = None
        self.y_min: Optional[float] = None
        self.y_max: Optional[float] = None

    # psycopg2 calls write() once per COPY row/message.
    def write(self, data: bytes) -> int:
        buf = self.buf
        buf.extend(data)
        if not self.header_done:
            if len(buf) < self.HEADER_FIXED:
                return len(data)
            if bytes(buf[:11]) != self.SIGNATURE:
                raise RuntimeError("Unexpected data from PostgreSQL: bad binary COPY header.")
            ext_len = int.from_bytes(buf[15:19], "big")
            if len(buf) < self.HEADER_FIXED + ext_len:
                return len(data)
            del buf[:self.HEADER_FIXED + ext_len]
            self.header_done = True
        if len(buf) >= self.threshold:
            self._drain()
        return len(data)

    @staticmethod
    def _merge_min(current: Optional[float], new: float) -> Optional[float]:
        if math.isnan(new):
            return current
        return new if current is None or new < current else current

    @staticmethod
    def _merge_max(current: Optional[float], new: float) -> Optional[float]:
        if math.isnan(new):
            return current
        return new if current is None or new > current else current

    def _drain(self) -> None:
        count = len(self.buf) // self.itemsize
        if count == 0:
            return
        raw = bytes(self.buf[:count * self.itemsize])
        del self.buf[:count * self.itemsize]
        arr = np.frombuffer(raw, dtype=self.dtype)
        if not ((arr["n"] == 2).all() and (arr["lx"] == self.field_len).all()
                and (arr["ly"] == self.field_len).all()):
            raise RuntimeError("Unexpected binary COPY row layout (NULL or non-float column?).")
        x = arr["x"].astype(self.native)  # big-endian -> native
        y = arr["y"].astype(self.native)
        # Same behaviour as before: NaN / +-inf are treated as gaps (NaN).
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.nan
        bad = ~np.isfinite(y)
        if bad.any():
            y[bad] = np.nan
        self.x_min = self._merge_min(self.x_min, float(np.fmin.reduce(x)))
        self.x_max = self._merge_max(self.x_max, float(np.fmax.reduce(x)))
        self.y_min = self._merge_min(self.y_min, float(np.fmin.reduce(y)))
        self.y_max = self._merge_max(self.y_max, float(np.fmax.reduce(y)))
        self.xs.append(x)
        self.ys.append(y)
        self.rows += count
        if self.on_block is not None:
            self.on_block(self.rows)

    def finish(self) -> None:
        self._drain()
        if bytes(self.buf) != self.TRAILER:
            raise RuntimeError("Binary COPY stream ended unexpectedly; please retry.")
        self.buf.clear()


def _raster_chunk(sample_range: tuple[float, float], value_range: tuple[float, float],
                  first_sample: int, values: np.ndarray, previous: Optional[float]):
    """Rasterize one block as a polyline (runs in a worker thread)."""
    n = len(values)
    if previous is None:
        samples = np.arange(first_sample, first_sample + n, dtype=np.float64)
        v = values.astype(np.float64)
    else:
        # Carry the previous block's final vertex so the segment joining the
        # two blocks is also rasterized.
        samples = np.arange(first_sample - 1, first_sample + n, dtype=np.float64)
        v = np.empty(n + 1, dtype=np.float64)
        v[0] = previous
        v[1:] = values
    frame = pd.DataFrame({"sample": samples, "v": v})
    canvas = ds.Canvas(plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT,
                       x_range=sample_range, y_range=value_range)
    return canvas.line(frame, x="sample", y="v", agg=ds.count())


def _shade(template, accum: np.ndarray) -> np.ndarray:
    if template is not None and np.any(accum):
        shaded = template.copy(data=accum)
        return np.asarray(tf.shade(shaded, cmap=[LINE_COLOUR], how="eq_hist", alpha=255).to_pil())
    return np.zeros((RASTER_HEIGHT, RASTER_WIDTH, 4), dtype=np.uint8)


class DatabaseRasterWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object, tuple, tuple, int, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings

    @Slot()
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
            pg_type = sql.SQL("float4" if TRANSFER_FLOAT32 else "float8")

            # NULL -> NaN in SQL keeps every row the same fixed size.
            query = sql.SQL(
                "SELECT COALESCE({x}::{t}, 'NaN'::{t}), COALESCE({y}::{t}, 'NaN'::{t}) "
                "FROM {relation} WHERE {train} = %s"
            ).format(t=pg_type, relation=relation, **ids)
            if s.order_column:
                query += sql.SQL(" ORDER BY {} ASC NULLS LAST").format(
                    checked_identifier(s.order_column, "order_by/order_column")
                )

            # ---------------- 1. download (binary COPY) ----------------
            dl_started = time.perf_counter()

            def on_block(rows: int) -> None:
                rate = rows / max(time.perf_counter() - dl_started, 1e-9)
                self.progress.emit(
                    f"Downloading Train 3 rows… {rows:,} ({rate / 1e6:.2f} M rows/s)"
                )

            parser = BinaryCopyParser(TRANSFER_FLOAT32, DOWNLOAD_BLOCK_ROWS, on_block)
            self.progress.emit("Connecting to PostgreSQL…")
            connection = psycopg2.connect(**s.connection)
            try:
                with connection.cursor() as cur:
                    inner = cur.mogrify(query.as_string(connection), (s.train_value,)).decode("utf-8")
                    self.progress.emit("Streaming Train 3 rows…")
                    cur.copy_expert(f"COPY ({inner}) TO STDOUT WITH (FORMAT BINARY)", parser)
            finally:
                connection.close()
            parser.finish()
            total = parser.rows
            if total == 0:
                raise RuntimeError(f"No rows found where {s.train_column} = {s.train_value!r}.")
            dl_time = time.perf_counter() - dl_started

            x_range = finite_range(parser.x_min, parser.x_max)
            y_range = finite_range(parser.y_min, parser.y_max)
            sample_range = (0.5, float(total) + 0.5)

            # ---------------- 2. parallel rasterization ----------------
            ras_started = time.perf_counter()
            self.progress.emit(f"Rasterizing all {total:,} Train 3 rows… 0%")
            offsets = [0]
            for block in parser.xs:
                offsets.append(offsets[-1] + len(block))

            accum = [np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=np.uint32),
                     np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=np.uint32)]
            templates: list[Any] = [None, None]
            with ThreadPoolExecutor(max_workers=RASTER_WORKERS) as pool:
                futures = {}
                for axis, (blocks, rng) in enumerate(((parser.xs, x_range), (parser.ys, y_range))):
                    for i, values in enumerate(blocks):
                        previous = float(blocks[i - 1][-1]) if i else None
                        fut = pool.submit(_raster_chunk, sample_range, rng,
                                          offsets[i] + 1, values, previous)
                        futures[fut] = axis
                done = 0
                for fut in as_completed(futures):
                    axis = futures[fut]
                    result = fut.result()
                    templates[axis] = result
                    # Counts can be added because every block uses identical
                    # ranges and dimensions.
                    np.add(accum[axis], np.asarray(result.values, dtype=np.uint32), out=accum[axis])
                    done += 1
                    if done % 4 == 0 or done == len(futures):
                        self.progress.emit(
                            f"Rasterizing all {total:,} Train 3 rows… {done * 100.0 / len(futures):.0f}%"
                        )
            ras_time = time.perf_counter() - ras_started

            # ---------------- 3. shade ----------------
            self.progress.emit("Finishing X_g and Y_g images…")
            x_rgba = _shade(templates[0], accum[0])
            y_rgba = _shade(templates[1], accum[1])
            elapsed = time.perf_counter() - started
            print(f"Timing: download {dl_time:.1f}s, rasterize {ras_time:.1f}s, "
                  f"total {elapsed:.1f}s ({total:,} rows, {RASTER_WORKERS} raster threads).")
            self.completed.emit(
                np.ascontiguousarray(np.flipud(x_rgba)),
                np.ascontiguousarray(np.flipud(y_rgba)),
                x_range, y_range, total, elapsed,
            )
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()


_APP_INSTANCE: Optional[QApplication] = None


def ensure_application() -> QApplication:
    """Create/reuse a widget application and retain it for the window lifetime."""
    global _APP_INSTANCE
    app = QCoreApplication.instance()
    if app is not None and not isinstance(app, QApplication):
        raise RuntimeError(
            "A non-widget Qt application already exists. "
            "Run dstrain.py in a fresh Python process."
        )
    if app is not None and QThread.currentThread() != app.thread():
        raise RuntimeError("Create the viewer on the QApplication's GUI thread.")
    if app is None:
        app = QApplication(sys.argv)
    _APP_INSTANCE = app
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    return app


class Train3Window(QMainWindow):
    def __init__(self, settings: Optional[Settings], config_error: Optional[str] = None) -> None:
        ensure_application()
        super().__init__()
        self._close_pending = False
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
            # quit() is thread-safe. A direct connection also permits cleanup
            # if the GUI event loop has already stopped.
            self.worker.finished.connect(self.thread.quit, Qt.DirectConnection)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self._worker_finished)
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

    @Slot(str)
    def _set_message(self, text: str) -> None:
        self.x_message.setText(text)
        self.y_message.setText(text)
        # Keep the status near the middle while the graph has not been drawn.
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)

    @Slot(object, object, tuple, tuple, int, float)
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

    @Slot(str)
    def _show_error(self, message: str) -> None:
        msg = "Could not load Train 3 data\n\n" + message
        self.x_message.setText(msg)
        self.y_message.setText(msg)
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)
        print(msg, file=sys.stderr)

    @Slot()
    def _worker_finished(self) -> None:
        if self._close_pending:
            self.close()

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            # quit() cannot interrupt a running COPY or raster function. Keep
            # the window/thread alive and the GUI responsive until work ends.
            self._close_pending = True
            self.setWindowTitle("Closing after the current data operation finishes…")
            event.ignore()
            return
        if self.thread is not None:
            self.thread.wait()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Path to db_config.json (default: beside this script)")
    parser.add_argument("--check-gui", action="store_true",
                        help="Open both plots without database access, then close")
    args = parser.parse_args()
    faulthandler.enable()
    print(f"Script: {Path(__file__).resolve()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"PySide6: {PySide6.__version__}; pyqtgraph: {pg.__version__}; "
          f"Qt binding: {pg.Qt.QT_LIB}", flush=True)
    print(f"Configuration: {args.config.resolve()}", flush=True)
    app = ensure_application()
    print("QApplication ready; constructing the viewer…", flush=True)
    if args.check_gui:
        settings, error = None, None
    else:
        try:
            settings = read_settings(args.config)
            error = None
        except Exception as exc:
            settings = None
            error = f"Configuration error: {exc}"
    window = Train3Window(settings, error)
    window.show()
    if args.check_gui:
        window._set_message("GUI check passed — no database connection requested.")
        app.processEvents()
        print("GUI CHECK PASSED: main window and both plots created.", flush=True)
        QTimer.singleShot(1000, window.close)
    try:
        return app.exec()
    finally:
        # Also protect against programmatic app.quit(), which bypasses closeEvent.
        if window.thread is not None:
            window.thread.quit()
            window.thread.wait()


if __name__ == "__main__":
    raise SystemExit(main())
