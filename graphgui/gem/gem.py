#!/usr/bin/env python3
"""Train 3 viewer: Ultra-fast in-memory downsampling + Datashader rendering.

Install: python -m pip install PySide6 pyqtgraph psycopg2-binary pandas datashader
Run: python dstrain_fast.py
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
import warnings
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

STARTED = time.perf_counter()

import numpy as np
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
DOWNLOAD_BLOCK_ROWS = 500_000
DOWNSAMPLE_FACTOR = 250  # Reduces 28.8M rows to ~230k points dynamically


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
        raise FileNotFoundError(f"Cannot find configuration: {path.resolve()}. Put db_config.json there.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    
    connection_source = raw.get("connection", raw)
    allowed = {
        "host", "port", "dbname", "database", "user", "password",
        "connect_timeout", "application_name", "sslmode", "sslrootcert",
        "sslcert", "sslkey", "options", "keepalives", "keepalives_idle",
        "keepalives_interval", "keepalives_count",
    }
    conn = {k: v for k, v in connection_source.items() if k in allowed}
    conn["dbname"] = connection_source.get("dbname", connection_source.get("database", "ofc_daqhat_test"))
    conn["port"] = int(conn.get("port", 5432))
    conn.setdefault("connect_timeout", 15)
    conn.setdefault("application_name", "train3_datashader_viewer")

    data = raw.get("data", {})
    schema = data.get("schema", raw.get("schema", "public"))
    table = data.get("table", raw.get("table", "test_adxl"))
    train_column = data.get("train_column", raw.get("train_column", "train_no"))
    train_value = data.get("train_value", raw.get("train_no", 3))
    x_column = data.get("x_column", raw.get("x_column", "x_g"))
    y_column = data.get("y_column", raw.get("y_column", "y_g"))
    
    if "order_column" in data: order_column = data["order_column"]
    elif "order_by" in raw: order_column = raw["order_by"]
    else: order_column = "timestamp"

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
    pad = (hi - lo) * 0.005
    return lo - pad, hi + pad


class BinaryCopyParser:
    """File-like sink for psycopg2. Performs strictly in-memory Min-Max Decimation."""
    SIGNATURE = b"PGCOPY\n\xff\r\n\x00"
    HEADER_FIXED = 19
    TRAILER = b"\xff\xff"

    def __init__(self, float32: bool, block_rows: int, on_block: Optional[Callable[[int], None]] = None) -> None:
        ftype = ">f4" if float32 else ">f8"
        self.field_len = 4 if float32 else 8
        self.native = np.float32 if float32 else np.float64
        self.dtype = np.dtype([("n", ">i2"), ("lx", ">i4"), ("x", ftype), ("ly", ">i4"), ("y", ftype)])
        self.itemsize = self.dtype.itemsize
        self.threshold = block_rows * self.itemsize
        self.on_block = on_block
        
        self.buf = bytearray()
        self.header_done = False
        self.xs: list[np.ndarray] = []
        self.ys: list[np.ndarray] = []
        self.idxs: list[np.ndarray] = []
        
        self.rows = 0
        self.x_min, self.x_max = None, None
        self.y_min, self.y_max = None, None

    def write(self, data: bytes) -> int:
        buf = self.buf
        buf.extend(data)
        if not self.header_done:
            if len(buf) < self.HEADER_FIXED: return len(data)
            ext_len = int.from_bytes(buf[15:19], "big")
            if len(buf) < self.HEADER_FIXED + ext_len: return len(data)
            del buf[:self.HEADER_FIXED + ext_len]
            self.header_done = True
        if len(buf) >= self.threshold:
            self._drain()
        return len(data)

    def _decimate(self, arr: np.ndarray, start_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Min-Max decimation to dramatically reduce vertex count without altering the visual envelope."""
        factor = DOWNSAMPLE_FACTOR
        remainder = len(arr) % factor
        
        main_part = arr[:-remainder] if remainder else arr
        tail = arr[-remainder:] if remainder else np.empty(0, dtype=arr.dtype)
        
        num_blocks = len(main_part) // factor

        if num_blocks > 0:
            reshaped = main_part.reshape((num_blocks, factor))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mins = np.nanmin(reshaped, axis=1)
                maxs = np.nanmax(reshaped, axis=1)

            ds_arr = np.empty(num_blocks * 2, dtype=arr.dtype)
            ds_arr[0::2] = mins
            ds_arr[1::2] = maxs

            ds_idx = np.empty(num_blocks * 2, dtype=np.float64)
            base_idx = start_idx + np.arange(num_blocks, dtype=np.float64) * factor
            ds_idx[0::2] = base_idx
            ds_idx[1::2] = base_idx + (factor - 1)
        else:
            ds_arr = np.empty(0, dtype=arr.dtype)
            ds_idx = np.empty(0, dtype=np.float64)

        if len(tail) > 0:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                tail_arr = np.array([np.nanmin(tail), np.nanmax(tail)], dtype=arr.dtype)
            tail_idx = np.array([start_idx + len(main_part), start_idx + len(arr) - 1], dtype=np.float64)
            ds_arr = np.concatenate([ds_arr, tail_arr])
            ds_idx = np.concatenate([ds_idx, tail_idx])

        return ds_arr, ds_idx

    def _drain(self) -> None:
        count = len(self.buf) // self.itemsize
        if count == 0: return
        raw = bytes(self.buf[:count * self.itemsize])
        del self.buf[:count * self.itemsize]
        
        arr = np.frombuffer(raw, dtype=self.dtype)
        x = arr["x"].astype(self.native)
        y = arr["y"].astype(self.native)
        
        x[~np.isfinite(x)] = np.nan
        y[~np.isfinite(y)] = np.nan
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            self.x_min = np.nanmin([self.x_min, np.nanmin(x)]) if self.x_min is not None else np.nanmin(x)
            self.x_max = np.nanmax([self.x_max, np.nanmax(x)]) if self.x_max is not None else np.nanmax(x)
            self.y_min = np.nanmin([self.y_min, np.nanmin(y)]) if self.y_min is not None else np.nanmin(y)
            self.y_max = np.nanmax([self.y_max, np.nanmax(y)]) if self.y_max is not None else np.nanmax(y)

        ds_x, idx_x = self._decimate(x, self.rows + 1)
        ds_y, _ = self._decimate(y, self.rows + 1)

        self.xs.append(ds_x)
        self.ys.append(ds_y)
        self.idxs.append(idx_x)

        self.rows += count
        if self.on_block:
            self.on_block(self.rows)

    def finish(self) -> None:
        self._drain()
        self.buf.clear()


def coverage_rgba(mask):
    image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    if np.any(mask):
        image[..., :3] = 255
        image[..., 3] = mask.astype(np.uint8) * 255
    return image


def save_snapshot(path, images, ranges, total, ds_total, label):
    path = Path(path).resolve()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(str(temporary), x=images[0], y=images[1],
                            ranges=np.asarray(ranges), total=np.int64(total), ds_total=np.int64(ds_total),
                            captured=np.asarray(stamp), label=np.asarray(label), version=np.int64(1))
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as z:
        images = [z["x"], z["y"]]
        ranges = z["ranges"]
        total = int(z["total"])
        ds_total = int(z["ds_total"])
        label = f"SAVED SNAPSHOT: {str(z['label'])} (not refreshed)"
    return images, [tuple(r) for r in ranges], total, ds_total, label


class DatabaseRasterWorker(QObject):
    progress = Signal(str)
    completed = Signal(object, object, tuple, tuple, int, int, float)
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
                images, ranges, total, ds_total, self.display_label = load_snapshot(self.options.snapshot)
            else:
                images, ranges, total, ds_total = self._load_fresh()
                if self.options.save_snapshot:
                    save_snapshot(self.options.save_snapshot, images, ranges, total, ds_total, self.display_label)
            self.completed.emit(images[0], images[1], ranges[0], ranges[1], total, ds_total, time.perf_counter() - started)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def _load_fresh(self):
        global ds, pd, psycopg2
        self.progress.emit("Loading database driver…")
        import psycopg2 as driver; from psycopg2 import sql; psycopg2 = driver
        
        s = self.settings
        self.display_label = f"Train {s.train_value} — X_g / Y_g"
        ids = {key: sql.Identifier(checked_identifier(value, key)) for key, value in (
            ("schema", s.schema), ("table", s.table), ("train", s.train_column), ("x", s.x_column), ("y", s.y_column))}
        relation = sql.SQL("{}.{}").format(ids["schema"], ids["table"])
        pg_type = sql.SQL("float8" if self.options.float64 else "float4")
        
        query = sql.SQL("SELECT COALESCE({x}::{t}, 'NaN'::{t}), COALESCE({y}::{t}, 'NaN'::{t}) FROM {relation} WHERE {train} = %s").format(t=pg_type, relation=relation, **ids)
        if s.order_column: query += sql.SQL(" ORDER BY {} ASC NULLS LAST").format(sql.Identifier(s.order_column))

        download_start = time.perf_counter()
        def on_block(rows):
            self.progress.emit(f"Downloading stream: {rows:,} rows ...")

        binary = BinaryCopyParser(not self.options.float64, self.options.chunk_rows, on_block)
        
        self.progress.emit("Executing ordered database query...")
        connection = driver.connect(**s.connection)
        try:
            with connection.cursor() as cur:
                inner = cur.mogrify(query.as_string(connection), (s.train_value,)).decode(driver.extensions.encodings[connection.encoding])
                cur.copy_expert(f"COPY ({inner}) TO STDOUT WITH (FORMAT BINARY)", binary)
        finally:
            connection.close()
        binary.finish()
        
        total = binary.rows
        if not total: raise RuntimeError(f"No rows found for train {s.train_value!r}")
        
        # Merge downsampled chunks in RAM instantly 
        ds_x = np.concatenate(binary.xs)
        ds_y = np.concatenate(binary.ys)
        ds_idx = np.concatenate(binary.idxs)
        ds_total = len(ds_idx)

        msg = f"Downsampled {total:,} rows to {ds_total:,} points ({total/ds_total:.1f}x reduction). Rendering..."
        self.progress.emit(msg)
        print(msg, flush=True)

        ranges = [finite_range(binary.x_min, binary.x_max), finite_range(binary.y_min, binary.y_max)]
        sample_range = (0.5, float(total) + 0.5)

        import datashader as datashader_module
        import pandas as pandas_module
        ds, pd = datashader_module, pandas_module

        # Render exactly once via Datashader
        canvas_x = ds.Canvas(plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT, x_range=sample_range, y_range=ranges[0])
        canvas_y = ds.Canvas(plot_width=RASTER_WIDTH, plot_height=RASTER_HEIGHT, x_range=sample_range, y_range=ranges[1])

        df_x = pd.DataFrame({"sample": ds_idx, "v": ds_x})
        df_y = pd.DataFrame({"sample": ds_idx, "v": ds_y})

        mask_x = np.asarray(canvas_x.line(df_x, x="sample", y="v", agg=ds.any()).values)
        mask_y = np.asarray(canvas_y.line(df_y, x="sample", y="v", agg=ds.any()).values)

        return [coverage_rgba(mask_x), coverage_rgba(mask_y)], ranges, total, ds_total


_APP_INSTANCE: Optional[QApplication] = None

def ensure_application() -> QApplication:
    global _APP_INSTANCE
    if QCoreApplication.instance() is None:
        _APP_INSTANCE = QApplication(sys.argv)
    else:
        _APP_INSTANCE = QCoreApplication.instance()
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    return _APP_INSTANCE


class Train3Window(QMainWindow):
    def __init__(self, settings, config_error=None, options=None) -> None:
        ensure_application()
        super().__init__()
        self._close_pending = False
        self.setWindowTitle("Train 3 — X_g and Y_g")
        self.resize(1400, 760)
        self.setStyleSheet("QMainWindow { background: #f4f5f6; }")
        
        root = QWidget()
        layout = QHBoxLayout(root)
        self.setCentralWidget(root)
        self.x_plot, self.x_message, self.x_image = self._make_plot("X_g")
        self.y_plot, self.y_message, self.y_image = self._make_plot("Y_g")
        layout.addWidget(self.x_plot, 1)
        layout.addWidget(self.y_plot, 1)

        if config_error:
            self._show_error(config_error)
        elif settings is not None or (options and options.snapshot):
            self.thread = QThread(self)
            self.worker = DatabaseRasterWorker(settings, options)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run, Qt.DirectConnection)
            self.worker.progress.connect(self._set_message)
            self.worker.completed.connect(self._show_images)
            self.worker.failed.connect(self._show_error)
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
        message = pg.TextItem("Loading…", color="white", anchor=(0.5, 0.5))
        plot.addItem(message)
        image = pg.ImageItem()
        plot.addItem(image)
        return plot, message, image

    @Slot(str)
    def _set_message(self, text: str) -> None:
        self.statusBar().showMessage(text)
        self.x_message.setText(text)
        self.y_message.setText(text)

    @Slot(object, object, tuple, tuple, int, int, float)
    def _show_images(self, x_rgba, y_rgba, x_range, y_range, total, ds_total, elapsed):
        if x_rgba.size:
            self.x_image.setImage(x_rgba, autoLevels=False, levels=None)
            self.x_image.setRect(QRectF(0.5, x_range[0], float(total), x_range[1] - x_range[0]))
        if y_rgba.size:
            self.y_image.setImage(y_rgba, autoLevels=False, levels=None)
            self.y_image.setRect(QRectF(0.5, y_range[0], float(total), y_range[1] - y_range[0]))
        
        for p, r in ((self.x_plot, x_range), (self.y_plot, y_range)):
            p.setXRange(0.5, float(total) + 0.5, padding=0)
            p.setYRange(r[0], r[1], padding=0)
            
        self.x_plot.removeItem(self.x_message)
        self.y_plot.removeItem(self.y_message)
        
        label = self.worker.display_label if self.worker else "Train 3"
        self.setWindowTitle(f"{label} · {total:,} rows (Downsampled to {ds_total:,}) · Datashader")
        self.statusBar().showMessage(f"Loaded {total:,} rows | Downsampled to {ds_total:,} points | Worker: {elapsed:.1f}s")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        msg = "Could not load Train 3 data\n\n" + message
        self.x_message.setText(msg)
        self.y_message.setText(msg)
        print(msg, file=sys.stderr)

    @Slot()
    def _worker_finished(self) -> None:
        if self._close_pending: self.close()

    def closeEvent(self, event) -> None:
        if self.thread is not None and self.thread.isRunning():
            self._close_pending = True
            self.setWindowTitle("Closing after data operation finishes…")
            event.ignore()
            return
        if self.thread: self.thread.wait()
        event.accept()

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check-gui", action="store_true")
    parser.add_argument("--workers", type=int, help="Legacy option (ignored - optimized for RAM)")
    parser.add_argument("--chunk-rows", type=int, default=DOWNLOAD_BLOCK_ROWS)
    parser.add_argument("--spool-dir", type=Path, help="Legacy option (ignored - optimized for RAM)")
    parser.add_argument("--float64", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--save-snapshot", type=Path)
    args = parser.parse_args()

    app = ensure_application()
    
    settings, error = None, None
    if not args.check_gui and not args.snapshot:
        try: settings = read_settings(args.config)
        except Exception as exc: error = f"Configuration error: {exc}"
            
    window = Train3Window(settings, error, args)
    window.show()
    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())