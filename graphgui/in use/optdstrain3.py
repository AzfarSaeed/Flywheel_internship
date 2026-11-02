#!/usr/bin/env python3
"""Train 3 viewer: full-data Datashader rendering with bounded RAM use.

Install: python -m pip install PySide6 pyqtgraph psycopg2-binary pandas datashader
Run: python dstrain_fast.py
Test GUI only: python dstrain_fast.py --check-gui
Save a finished plot: python dstrain_fast.py --save-snapshot train3.npz
Reopen that saved plot (no database refresh): python dstrain_fast.py --snapshot train3.npz

All rows are downloaded in one ordered binary COPY and spooled to temporary
NumPy chunks. Ranges come from that same stream. A bounded queue rasterizes
all line segments, including chunk joins. NaNs remain gaps. The original
float32 transfer precision is retained; use --float64 for double precision.
Solid white occupancy rasters avoid unnecessary density/count/shading work.
The GUI appears before database/Datashader imports. Fresh full-data loading
cannot have a guaranteed deadline without measuring the server and network.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

STARTED = time.perf_counter()

import numpy as np
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

# Heavy libraries are imported in the worker after the window is visible.
ds = pd = psycopg2 = None
RASTER_WORKERS = min(2, os.cpu_count() or 1)
DOWNLOAD_BLOCK_ROWS = 500_000


def checked_identifier(value: Any, label: str):
    if not isinstance(value, str) or not value or any(ch not in VALID_IDENTIFIER for ch in value):
        raise ValueError(f"{label} must contain only letters, digits, and underscores.")
    return value


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
                 on_block: Optional[Callable[[int], None]] = None,
                 on_chunk=None) -> None:
        ftype = ">f4" if float32 else ">f8"
        self.field_len = 4 if float32 else 8
        self.native = np.float32 if float32 else np.float64
        self.dtype = np.dtype([
            ("n", ">i2"), ("lx", ">i4"), ("x", ftype), ("ly", ">i4"), ("y", ftype),
        ])
        self.itemsize = self.dtype.itemsize
        self.threshold = block_rows * self.itemsize
        self.on_block = on_block
        self.on_chunk = on_chunk
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
        if self.on_chunk is None:
            self.xs.append(x)
            self.ys.append(y)
        else:
            self.on_chunk(x, y)
        self.rows += count
        if self.on_block is not None:
            self.on_block(self.rows)

    def finish(self) -> None:
        self._drain()
        if bytes(self.buf) != self.TRAILER:
            raise RuntimeError("Binary COPY stream ended unexpectedly; please retry.")
        self.buf.clear()


def _raster_values(sample_range, value_range, first_sample, values, previous):
    """Visit every vertex; only the boolean pixel coverage is retained."""
    n = len(values)
    if previous is None:
        samples = np.arange(first_sample, first_sample + n, dtype=np.float64)
        v = values
    else:
        samples = np.arange(first_sample - 1, first_sample + n, dtype=np.float64)
        v = np.empty(n + 1, dtype=values.dtype)
        v[0] = previous
        v[1:] = values
    frame = pd.DataFrame({"sample": samples, "v": v}, copy=False)
    canvas = ds.Canvas(plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT,
                       x_range=sample_range, y_range=value_range)
    return np.asarray(canvas.line(frame, x="sample", y="v", agg=ds.any()).values)


def _raster_file(task, sample_range, ranges):
    axis, path, first_sample, previous = task
    values = np.load(path, allow_pickle=False)
    return axis, _raster_values(sample_range, ranges[axis], first_sample, values, previous)


def rasterize_tasks(tasks, sample_range, ranges, workers, progress):
    """Never retain more than workers pending/running raster results."""
    accum = [np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=bool) for _ in range(2)]
    iterator = iter(tasks)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        def submit_next():
            task = next(iterator, None)
            if task is not None:
                pending.add(pool.submit(_raster_file, task, sample_range, ranges))
        for _ in range(workers):
            submit_next()
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                axis, pixels = future.result()
                np.logical_or(accum[axis], pixels, out=accum[axis])
                done += 1
                progress(done, len(tasks))
                submit_next()
            # Release results before waiting on the next group.
            del ready, future, pixels
    return accum


def coverage_rgba(mask):
    image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    if np.any(mask):
        image[..., :3] = 255
        image[..., 3] = mask.astype(np.uint8) * 255
    return image


def save_snapshot(path, images, ranges, total, label):
    """Atomic local snapshot: no connection settings or passwords stored."""
    path = Path(path).resolve()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(str(temporary), x=images[0], y=images[1],
                            ranges=np.asarray(ranges), total=np.int64(total),
                            captured=np.asarray(stamp), label=np.asarray(label), version=np.int64(1))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as z:
        if int(z["version"]) != 1:
            raise ValueError("Unsupported snapshot version")
        images = [z["x"], z["y"]]
        ranges = z["ranges"]
        total = int(z["total"])
        label = f"SAVED SNAPSHOT: {str(z['label'])} — {str(z['captured'])} (not refreshed)"
    if total <= 0 or ranges.shape != (2, 2) or not np.isfinite(ranges).all():
        raise ValueError("Invalid snapshot row count or ranges")
    if not np.all(ranges[:, 1] > ranges[:, 0]):
        raise ValueError("Invalid snapshot bounds")
    if any(im.dtype != np.uint8 or im.shape != (RASTER_HEIGHT, RASTER_WIDTH, 4) for im in images):
        raise ValueError("Invalid snapshot image format")
    return images, [tuple(r) for r in ranges], total, label


class DatabaseRasterWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object, tuple, tuple, int, float)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, settings, options) -> None:
        super().__init__()
        self.settings = settings
        self.options = options
        self.display_label = "Train 3"

    @Slot()
    def run(self) -> None:
        started = time.perf_counter()
        try:
            if self.options.snapshot:
                self.progress.emit("Opening saved plot (no database refresh)…")
                images, ranges, total, self.display_label = load_snapshot(self.options.snapshot)
            else:
                with tempfile.TemporaryDirectory(prefix="train3-", dir=self.options.spool_dir) as folder:
                    images, ranges, total = self._load_fresh(Path(folder))
                if self.options.save_snapshot:
                    try:
                        save_snapshot(self.options.save_snapshot, images, ranges, total, self.display_label)
                        print(f"Saved snapshot: {self.options.save_snapshot.resolve()}", flush=True)
                    except Exception as exc:
                        # A cache-write failure must not discard a successful live render.
                        print(f"Plot loaded, but snapshot could not be saved: {exc}", file=sys.stderr, flush=True)
            self.completed.emit(images[0], images[1], ranges[0], ranges[1], total,
                                time.perf_counter() - started)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def _load_fresh(self, folder):
        global ds, pd, psycopg2
        self.progress.emit("Loading database driver…")
        t = time.perf_counter()
        import psycopg2 as driver
        from psycopg2 import sql
        psycopg2 = driver
        print(f"Database driver import: {time.perf_counter() - t:.2f}s", flush=True)
        s = self.settings
        self.display_label = f"Train {s.train_value} — X_g / Y_g"
        ids = {key: sql.Identifier(checked_identifier(value, key)) for key, value in (
            ("schema", s.schema), ("table", s.table), ("train", s.train_column),
            ("x", s.x_column), ("y", s.y_column))}
        relation = sql.SQL("{}.{}").format(ids["schema"], ids["table"])
        pg_type = sql.SQL("float8" if self.options.float64 else "float4")
        query = sql.SQL(
            "SELECT COALESCE({x}::{t}, 'NaN'::{t}), COALESCE({y}::{t}, 'NaN'::{t}) "
            "FROM {relation} WHERE {train} = %s"
        ).format(t=pg_type, relation=relation, **ids)
        if s.order_column:
            query += sql.SQL(" ORDER BY {} ASC NULLS LAST").format(sql.Identifier(s.order_column))
        tasks = []
        offset = 0
        previous = [None, None]
        def spool(x, y):
            nonlocal offset
            for axis, values in enumerate((x, y)):
                path = folder / f"{offset}-{axis}.npy"
                np.save(str(path), values, allow_pickle=False)
                tasks.append((axis, path, offset + 1, previous[axis]))
                previous[axis] = float(values[-1])
            offset += len(x)
        download_start = time.perf_counter()
        def on_block(rows):
            elapsed = time.perf_counter() - download_start
            self.progress.emit(f"Downloaded {rows:,} rows in {elapsed:.1f}s — all rows retained")
        binary = BinaryCopyParser(not self.options.float64, self.options.chunk_rows, on_block, spool)
        self.progress.emit("Connecting / waiting for the ordered database query…")
        connection = driver.connect(**s.connection)
        try:
            with connection.cursor() as cur:
                inner = cur.mogrify(query.as_string(connection), (s.train_value,)).decode(driver.extensions.encodings[connection.encoding])
                cur.copy_expert(f"COPY ({inner}) TO STDOUT WITH (FORMAT BINARY)", binary)
        finally:
            connection.close()
        binary.finish()
        total = binary.rows
        if not total:
            raise RuntimeError(f"No rows found for train {s.train_value!r}")
        print(f"Download + temporary chunk writes: {time.perf_counter() - download_start:.2f}s; {total:,} rows", flush=True)
        ranges = [finite_range(binary.x_min, binary.x_max), finite_range(binary.y_min, binary.y_max)]
        sample_range = (0.5, float(total) + 0.5)
        self.progress.emit("Loading Datashader / compiling the renderer (first use can take longer)…")
        t = time.perf_counter()
        import datashader as datashader_module
        import pandas as pandas_module
        ds, pd = datashader_module, pandas_module
        dtype = np.float64 if self.options.float64 else np.float32
        # Compile once before concurrent jobs, avoiding simultaneous cold compilation.
        _raster_values((0.5, 3.5), (-1.0, 1.0), 1, np.array([0, .5, 0], dtype=dtype), None)
        print(f"Datashader import + warm-up: {time.perf_counter() - t:.2f}s", flush=True)
        t = time.perf_counter()
        masks = rasterize_tasks(tasks, sample_range, ranges, self.options.workers,
                                lambda done, count: self.progress.emit(
                                    f"Rendering all {total:,} rows — {done}/{count} chunks ({100 * done / count:.0f}%)"))
        images = [coverage_rgba(mask) for mask in masks]
        print(f"Rasterize + image conversion: {time.perf_counter() - t:.2f}s; {self.options.workers} workers", flush=True)
        return images, ranges, total


_APP_INSTANCE: Optional[QApplication] = None


def ensure_application() -> QApplication:
    """Create/reuse a widget application and retain it for the window lifetime."""
    global _APP_INSTANCE
    app = QCoreApplication.instance()
    if app is not None and not isinstance(app, QApplication):
        raise RuntimeError(
            "A non-widget Qt application already exists. "
            "Run dstrain_fast.py in a fresh Python process."
        )
    if app is not None and QThread.currentThread() != app.thread():
        raise RuntimeError("Create the viewer on the QApplication's GUI thread.")
    if app is None:
        app = QApplication(sys.argv)
    _APP_INSTANCE = app
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    return app


class Train3Window(QMainWindow):
    def __init__(self, settings: Optional[Settings], config_error: Optional[str] = None, options=None) -> None:
        ensure_application()
        super().__init__()
        self._close_pending = False
        self.options = options
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
        elif settings is not None or (options is not None and options.snapshot):
            self.thread = QThread(self)
            self.worker = DatabaseRasterWorker(settings, options)
            self.worker.moveToThread(self.thread)
            # QThread emits started from its new worker thread. Explicit delivery
            # ensures this blocking function runs there, not in the GUI thread.
            self.thread.started.connect(self.worker.run, Qt.DirectConnection)
            self.worker.progress.connect(self._set_message)
            self.worker.completed.connect(self._show_images)
            self.worker.failed.connect(self._show_error)
            # quit() is thread-safe. A direct connection also permits cleanup
            # if the GUI event loop has already stopped.
            self.worker.finished.connect(self.thread.quit, Qt.DirectConnection)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self._worker_finished)
            QTimer.singleShot(0, self.thread.start)

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
        self.statusBar().showMessage(text)
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
        label = self.worker.display_label if self.worker else "Train 3"
        self.setWindowTitle(f"{label} · {total:,} rows · Datashader")
        self.statusBar().showMessage(f"{label} — all {total:,} rows rendered; {elapsed:.1f}s worker time")
        print(f"Displayed {total:,} rows; worker {elapsed:.1f}s; elapsed since startup marker {time.perf_counter() - STARTED:.1f}s.", flush=True)

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
    parser.add_argument("--workers", type=int, default=RASTER_WORKERS, help="Concurrent raster jobs, 1 to 8 (default: 2)")
    parser.add_argument("--chunk-rows", type=int, default=DOWNLOAD_BLOCK_ROWS, help="Rows per chunk (default: 500000)")
    parser.add_argument("--spool-dir", type=Path, help="Temporary chunk directory; use a local SSD")
    parser.add_argument("--float64", action="store_true", help="Transfer double precision instead of float32")
    parser.add_argument("--snapshot", type=Path, help="Open a saved plot; does NOT refresh database data")
    parser.add_argument("--save-snapshot", type=Path, help="Save a finished fresh plot for fast reopening")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1000 <= args.chunk_rows <= 2000000:
        parser.error("--workers must be 1..8; --chunk-rows must be 1000..2000000")
    if args.snapshot and args.save_snapshot:
        parser.error("Use either --snapshot or --save-snapshot, not both")
    faulthandler.enable()
    print(f"Script: {Path(__file__).resolve()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"PySide6: {PySide6.__version__}; pyqtgraph: {pg.__version__}; "
          f"Qt binding: {pg.Qt.QT_LIB}", flush=True)
    print(f"Configuration: {args.config.resolve()}", flush=True)
    app = ensure_application()
    print("QApplication ready; constructing the viewer…", flush=True)
    if args.check_gui or args.snapshot:
        settings, error = None, None
    else:
        try:
            settings = read_settings(args.config)
            error = None
        except Exception as exc:
            settings = None
            error = f"Configuration error: {exc}"
    # Do not load a snapshot during a GUI-only check.
    if args.check_gui:
        args.snapshot = None
    window = Train3Window(settings, error, args)
    window.show()
    app.processEvents()
    print(f"Window shown: {time.perf_counter() - STARTED:.2f}s after startup marker", flush=True)
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
