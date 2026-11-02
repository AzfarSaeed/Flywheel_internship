#!/usr/bin/env python3
"""Fast Train 3 X_g / Y_g viewer using PostgreSQL-side min/max downsampling.

The database aggregates every matching Train 3 row into horizontal buckets.
For each bucket it returns min/max X_g and min/max Y_g. That preserves each
bucket's signal envelope while sending only a small result to the Windows GUI.
No raw 28.8-million-row result set is downloaded to the client.

First run / refresh (fresh database aggregation):
    python dstrain_fast_optimized.py --refresh
Normal run (uses the local image cache when available):
    python dstrain_fast_optimized.py
Force a GUI-only test:
    python dstrain_fast_optimized.py --check-gui

The first database refresh still scans the matching rows on PostgreSQL and
its duration depends on server load, query plan, and network availability.
The saved local cache makes subsequent launches independent of the database.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Keep one Qt binding throughout the application.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")
from PySide6.QtCore import QObject, QThread, QTimer, Qt, QRectF, Signal, Slot
from PySide6.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QWidget
import pyqtgraph as pg

STARTED = time.perf_counter()
CONFIG_PATH = Path(__file__).resolve().with_name("db_config.json")
DEFAULT_CACHE = Path(__file__).resolve().with_name("train3_envelope_cache.npz")
RASTER_WIDTH = 1600
RASTER_HEIGHT = 700
PANEL_BLUE = "#176783"
VALID_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
CACHE_VERSION = 1


def checked_identifier(value: Any, label: str) -> str:
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


def read_settings(path: Path) -> Settings:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Put db_config.json beside this script or use --config.")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is invalid JSON: {exc.msg}.") from exc
    if not isinstance(raw, dict):
        raise ValueError("The JSON configuration must be an object.")

    source = raw.get("connection", raw)
    if not isinstance(source, dict):
        raise ValueError("'connection' must be a JSON object.")
    allowed = {
        "host", "port", "dbname", "database", "user", "password",
        "connect_timeout", "application_name", "sslmode", "sslrootcert",
        "sslcert", "sslkey", "options", "keepalives", "keepalives_idle",
        "keepalives_interval", "keepalives_count",
    }
    connection = {key: value for key, value in source.items() if key in allowed}
    connection["dbname"] = source.get("dbname", source.get("database", "ofc_daqhat_test"))
    for key in ("host", "dbname", "user", "password"):
        if not connection.get(key):
            raise ValueError(f"Missing '{key}' in {path.name}.")
    connection["port"] = int(connection.get("port", 5432))
    connection.setdefault("connect_timeout", 15)
    connection.setdefault("application_name", "train3_minmax_viewer")
    connection.setdefault("keepalives", 1)
    connection.setdefault("keepalives_idle", 30)
    connection.setdefault("keepalives_interval", 10)
    connection.setdefault("keepalives_count", 6)

    data = raw.get("data", {})
    if not isinstance(data, dict):
        raise ValueError("'data' must be a JSON object when supplied.")
    schema = data.get("schema", raw.get("schema", "public"))
    table = data.get("table", raw.get("table", "test_adxl"))
    train_column = data.get("train_column", raw.get("train_column", "train_no"))
    train_value = data.get("train_value", raw.get("train_no", 3))
    x_column = data.get("x_column", raw.get("x_column", "x_g"))
    y_column = data.get("y_column", raw.get("y_column", "y_g"))

    if "order_column" in data:
        order_column = data["order_column"]
    elif "order_by" in raw:
        order_column = raw["order_by"]
    else:
        # Null selects fast physical-page grouping; set this to timestamp/id for
        # a chronological order (which can require a database index).
        order_column = None
    if order_column is not None and (not isinstance(order_column, str) or not order_column):
        raise ValueError("order_by/order_column must be a column name or null.")

    for value, label in (
        (schema, "schema"), (table, "table"), (train_column, "train_column"),
        (x_column, "x_column"), (y_column, "y_column"),
    ):
        checked_identifier(value, label)
    if order_column is not None:
        checked_identifier(order_column, "order_by/order_column")

    return Settings(connection, schema, table, train_column, train_value,
                    x_column, y_column, order_column)


def finite_range(low: Any, high: Any) -> tuple[float, float]:
    try:
        lo, hi = float(low), float(high)
    except (TypeError, ValueError, OverflowError):
        return (-1.0, 1.0)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return (-1.0, 1.0)
    if lo == hi:
        pad = max(abs(lo) * 0.01, 1e-6)
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.005
    return (lo - pad, hi + pad)


def finite_value_expression(column_sql):
    # Composable SQL: converts NULL, NaN and +/-Infinity to NULL so MIN/MAX
    # use finite values only, without interpolating an unquoted identifier.
    from psycopg2 import sql
    return sql.SQL(
        "CASE WHEN abs({c}::double precision) < 'Infinity'::double precision "
        "THEN {c}::double precision ELSE NULL END"
    ).format(c=column_sql)


def build_bucket_query(settings: Settings, bucket_count: int):
    """Build one compact SQL aggregation, returning <= bucket_count result rows."""
    from psycopg2 import sql

    schema = sql.Identifier(settings.schema)
    table = sql.Identifier(settings.table)
    relation = sql.SQL("{}.{}").format(schema, table)
    train = sql.Identifier(settings.train_column)
    x_col = sql.SQL("t.{}").format(sql.Identifier(settings.x_column))
    y_col = sql.SQL("t.{}").format(sql.Identifier(settings.y_column))
    x_expr = finite_value_expression(x_col)
    y_expr = finite_value_expression(y_col)

    if settings.order_column:
        order_expr = sql.SQL("t.{} ASC NULLS LAST, t.ctid ASC").format(
            sql.Identifier(settings.order_column)
        )
        query = sql.SQL(
            "WITH bucketed AS ("
            " SELECT NTILE(%s) OVER (ORDER BY {order_expr}) AS bucket,"
            " {x_expr} AS xv, {y_expr} AS yv"
            " FROM {relation} AS t WHERE t.{train} = %s"
            ") "
            "SELECT bucket, COUNT(*)::bigint AS source_rows,"
            " COUNT(xv)::bigint AS x_valid, MIN(xv), MAX(xv),"
            " COUNT(yv)::bigint AS y_valid, MIN(yv), MAX(yv)"
            " FROM bucketed GROUP BY bucket ORDER BY bucket"
        ).format(order_expr=order_expr, x_expr=x_expr, y_expr=y_expr,
                 relation=relation, train=train)
        return query, (bucket_count, settings.train_value), "chronological"

    # Fast mode: use the tuple's physical heap page to form buckets. This avoids
    # sorting 28.8M rows. It follows physical storage order, not guaranteed time order.
    relation_name = f'"{settings.schema}"."{settings.table}"'
    query = sql.SQL(
        "WITH sizing AS ("
        " SELECT GREATEST(CEIL(pg_relation_size(%s::regclass)::numeric /"
        " current_setting('block_size')::numeric / %s)::bigint, 1) AS blocks_per_bucket"
        "), bucketed AS ("
        " SELECT LEAST(%s - 1,"
        " (split_part(trim(both '()' from t.ctid::text), ',', 1)::bigint /"
        " sizing.blocks_per_bucket)::integer)::integer AS bucket,"
        " {x_expr} AS xv, {y_expr} AS yv"
        " FROM {relation} AS t CROSS JOIN sizing"
        " WHERE t.{train} = %s"
        ") "
        "SELECT bucket, COUNT(*)::bigint AS source_rows,"
        " COUNT(xv)::bigint AS x_valid, MIN(xv), MAX(xv),"
        " COUNT(yv)::bigint AS y_valid, MIN(yv), MAX(yv)"
        " FROM bucketed GROUP BY bucket ORDER BY bucket"
    ).format(x_expr=x_expr, y_expr=y_expr, relation=relation, train=train)
    return query, (relation_name, bucket_count, bucket_count, settings.train_value), "physical-page"


def envelope_image(min_values: np.ndarray, max_values: np.ndarray,
                   height: int = RASTER_HEIGHT, width: int = RASTER_WIDTH):
    """Rasterize each bucket's min/max as a continuous horizontal envelope.

    Every source row contributes to the SQL MIN/MAX aggregates. Since the
    database returns at most ``width`` buckets, each bucket is expanded across
    its corresponding horizontal pixel span; this avoids blank columns when
    some physical-page buckets contain no Train 3 rows.
    """
    mins = np.asarray(min_values, dtype=np.float64)
    maxs = np.asarray(max_values, dtype=np.float64)
    valid = np.isfinite(mins) & np.isfinite(maxs)
    if not valid.any():
        return np.zeros((height, width, 4), dtype=np.uint8), (-1.0, 1.0), 0

    y_min = float(np.min(mins[valid]))
    y_max = float(np.max(maxs[valid]))
    y_range = finite_range(y_min, y_max)
    low, high = y_range
    span = high - low
    mask = np.zeros((height, width), dtype=bool)

    n = len(mins)
    retained = 0
    for i in range(n):
        if not valid[i]:
            continue
        # A DB bucket spans one or more adjacent raster columns. With the
        # default settings there is one bucket per pixel column.
        col_start = (i * width) // max(n, 1)
        col_end = ((i + 1) * width) // max(n, 1)
        col_end = max(col_start + 1, col_end)
        col_end = min(width, col_end)

        # Pixel row 0 is the top of the plot (largest acceleration).
        row_top = int(round((high - float(maxs[i])) / span * (height - 1)))
        row_bottom = int(round((high - float(mins[i])) / span * (height - 1)))
        row_top = max(0, min(height - 1, row_top))
        row_bottom = max(0, min(height - 1, row_bottom))
        if row_bottom < row_top:
            row_top, row_bottom = row_bottom, row_top
        mask[row_top:row_bottom + 1, col_start:col_end] = True

        retained += 1
        if mins[i] != maxs[i]:
            retained += 1

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask, 0:3] = 255
    rgba[mask, 3] = 255
    return np.ascontiguousarray(np.flipud(rgba)), y_range, retained


def report_count(original: int, retained: int) -> tuple[int, float]:
    removed = max(0, int(original) - int(retained))
    percentage = (100.0 * removed / original) if original else 0.0
    return removed, percentage


def save_cache(path: Path, result: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds")
    temp_path = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        temp_path,
        x_image=result["x_image"], y_image=result["y_image"],
        x_range=np.asarray(result["x_range"], dtype=np.float64),
        y_range=np.asarray(result["y_range"], dtype=np.float64),
        total=np.int64(result["total"]),
        x_valid=np.int64(result["x_valid"]), y_valid=np.int64(result["y_valid"]),
        x_retained=np.int64(result["x_retained"]), y_retained=np.int64(result["y_retained"]),
        captured=np.asarray(captured), version=np.int64(CACHE_VERSION),
    )
    os.replace(temp_path, path)


def load_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        if int(z["version"]) != CACHE_VERSION:
            raise ValueError("The saved graph cache has an unsupported version; use --refresh.")
        result = {
            "x_image": z["x_image"], "y_image": z["y_image"],
            "x_range": tuple(float(v) for v in z["x_range"]),
            "y_range": tuple(float(v) for v in z["y_range"]),
            "total": int(z["total"]), "x_valid": int(z["x_valid"]),
            "y_valid": int(z["y_valid"]), "x_retained": int(z["x_retained"]),
            "y_retained": int(z["y_retained"]), "captured": str(z["captured"]),
        }
    expected = (RASTER_HEIGHT, RASTER_WIDTH, 4)
    if result["x_image"].shape != expected or result["y_image"].shape != expected:
        raise ValueError("The saved graph cache has invalid image dimensions; use --refresh.")
    if result["total"] <= 0:
        raise ValueError("The saved graph cache is empty; use --refresh.")
    return result


class DatabaseWorker(QObject):
    progress = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, settings: Optional[Settings], args, use_cache: bool):
        super().__init__()
        self.settings = settings
        self.args = args
        self.use_cache = use_cache

    @Slot()
    def run(self):
        started = time.perf_counter()
        try:
            if self.use_cache:
                self.progress.emit("Opening local graph cache…")
                try:
                    result = load_cache(self.args.cache_file)
                    result["source"] = "CACHED"
                except Exception as cache_error:
                    if self.settings is None:
                        raise RuntimeError(f"Could not open cache: {cache_error}")
                    self.progress.emit(f"Cache unavailable ({cache_error}); refreshing from database…")
                    result = self.load_database()
            else:
                result = self.load_database()

            result["elapsed"] = time.perf_counter() - started
            if result.get("source") == "DATABASE" and not self.args.no_cache:
                try:
                    self.progress.emit("Saving local cache…")
                    save_cache(self.args.cache_file, result)
                    print(f"Saved graph cache: {self.args.cache_file.resolve()}", flush=True)
                except Exception as exc:
                    print(f"Warning: could not save graph cache: {exc}", file=sys.stderr, flush=True)
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.finished.emit()

    def load_database(self) -> dict[str, Any]:
        from psycopg2 import connect

        if self.settings is None:
            raise RuntimeError("Database settings are unavailable. Check db_config.json.")
        s = self.settings
        query, params, method = build_bucket_query(s, self.args.buckets)
        self.progress.emit("Connecting to PostgreSQL…")
        t0 = time.perf_counter()
        connection = connect(**s.connection)
        try:
            self.progress.emit(f"Database is aggregating every Train {s.train_value} row into {self.args.buckets:,} buckets…")
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = cursor.fetchall()  # At most --buckets rows, not 28.8 million.
        finally:
            connection.close()
        query_elapsed = time.perf_counter() - t0
        if not rows:
            raise RuntimeError(f"No rows found where {s.train_column} = {s.train_value!r}.")

        total = sum(int(row[1]) for row in rows)
        x_valid = sum(int(row[2]) for row in rows)
        y_valid = sum(int(row[5]) for row in rows)
        x_min = np.asarray([np.nan if row[3] is None else float(row[3]) for row in rows], dtype=np.float64)
        x_max = np.asarray([np.nan if row[4] is None else float(row[4]) for row in rows], dtype=np.float64)
        y_min = np.asarray([np.nan if row[6] is None else float(row[6]) for row in rows], dtype=np.float64)
        y_max = np.asarray([np.nan if row[7] is None else float(row[7]) for row in rows], dtype=np.float64)

        self.progress.emit("Rendering the two min/max envelopes…")
        x_image, x_range, x_retained = envelope_image(x_min, x_max)
        y_image, y_range, y_retained = envelope_image(y_min, y_max)
        result = {
            "x_image": x_image, "y_image": y_image,
            "x_range": x_range, "y_range": y_range,
            "total": total, "x_valid": x_valid, "y_valid": y_valid,
            "x_retained": x_retained, "y_retained": y_retained,
            "bucket_rows": len(rows), "method": method,
            "query_elapsed": query_elapsed, "source": "DATABASE",
        }
        return result


class MainWindow(QMainWindow):
    def __init__(self, settings: Optional[Settings], config_error: Optional[str], args, use_cache: bool):
        super().__init__()
        self.args = args
        self.thread: Optional[QThread] = None
        self.worker: Optional[DatabaseWorker] = None
        self.close_pending = False
        self.setWindowTitle("Train 3 — X_g and Y_g")
        self.resize(1500, 780)
        self.setStyleSheet("QMainWindow { background: #f4f5f6; }")
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        self.setCentralWidget(root)
        self.x_plot, self.x_message, self.x_image = self.make_plot("X_g")
        self.y_plot, self.y_message, self.y_image = self.make_plot("Y_g")
        layout.addWidget(self.x_plot, 1)
        layout.addWidget(self.y_plot, 1)

        if args.check_gui:
            self.set_status("GUI check passed — no database connection requested.")
            QTimer.singleShot(1200, self.close)
        elif config_error and not use_cache:
            self.show_error(config_error)
        else:
            self.thread = QThread(self)
            self.worker = DatabaseWorker(settings, args, use_cache)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.worker.run, Qt.DirectConnection)
            self.worker.progress.connect(self.set_status)
            self.worker.completed.connect(self.show_result)
            self.worker.failed.connect(self.show_error)
            self.worker.finished.connect(self.thread.quit, Qt.DirectConnection)
            self.thread.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.worker_finished)
            QTimer.singleShot(0, self.thread.start)

    @staticmethod
    def make_plot(title: str):
        plot = pg.PlotWidget(background=PANEL_BLUE)
        plot.setMenuEnabled(False)
        plot.setMouseEnabled(x=False, y=False)
        plot.showGrid(x=False, y=False)
        plot.setTitle(title, color="white", size="16pt")
        plot.getPlotItem().hideAxis("left")
        plot.getPlotItem().hideAxis("bottom")
        plot.getPlotItem().setContentsMargins(0, 0, 0, 0)
        message = pg.TextItem("Loading…", color="white", anchor=(0.5, 0.5))
        plot.addItem(message)
        message.setPos(0.5, 0.5)
        image = pg.ImageItem()
        plot.addItem(image)
        return plot, message, image

    @Slot(str)
    def set_status(self, text: str):
        self.statusBar().showMessage(text)
        self.x_message.setText(text)
        self.y_message.setText(text)
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)

    @Slot(object)
    def show_result(self, result: dict):
        self.x_image.setImage(result["x_image"], autoLevels=False, levels=None)
        self.y_image.setImage(result["y_image"], autoLevels=False, levels=None)
        total = result["total"]
        self.x_image.setRect(QRectF(0.5, result["x_range"][0], float(total),
                                    result["x_range"][1] - result["x_range"][0]))
        self.y_image.setRect(QRectF(0.5, result["y_range"][0], float(total),
                                    result["y_range"][1] - result["y_range"][0]))
        self.x_plot.setXRange(0.5, float(total) + 0.5, padding=0)
        self.y_plot.setXRange(0.5, float(total) + 0.5, padding=0)
        self.x_plot.setYRange(*result["x_range"], padding=0)
        self.y_plot.setYRange(*result["y_range"], padding=0)
        self.x_plot.removeItem(self.x_message)
        self.y_plot.removeItem(self.y_message)

        x_removed, x_pct = report_count(result["x_valid"], result["x_retained"])
        y_removed, y_pct = report_count(result["y_valid"], result["y_retained"])
        source = result.get("source", "DATABASE")
        self.setWindowTitle(f"Train 3 — X_g / Y_g · {total:,} source rows · {source}")
        status = (
            f"{source}: {total:,} source rows | "
            f"X_g kept {result['x_retained']:,}, downsampled {x_removed:,} ({x_pct:.3f}%) | "
            f"Y_g kept {result['y_retained']:,}, downsampled {y_removed:,} ({y_pct:.3f}%)"
        )
        if result.get("method"):
            status += f" | grouping: {result['method']}"
        if result.get("query_elapsed") is not None:
            status += f" | database query: {result['query_elapsed']:.1f}s"
        if result.get("captured"):
            status += f" | cache saved: {result['captured']}"
        self.set_status(status)
        print("\nTrain 3 plot summary", flush=True)
        print(f"Source rows: {total:,}", flush=True)
        print(f"X_g: valid={result['x_valid']:,}; envelope boundaries kept={result['x_retained']:,}; "
              f"downsampled={x_removed:,} ({x_pct:.3f}%)", flush=True)
        print(f"Y_g: valid={result['y_valid']:,}; envelope boundaries kept={result['y_retained']:,}; "
              f"downsampled={y_removed:,} ({y_pct:.3f}%)", flush=True)
        if result.get("method"):
            print(f"Grouping mode: {result['method']}; returned buckets: {result.get('bucket_rows', 0):,}", flush=True)
        print(f"Worker elapsed: {result.get('elapsed', 0):.1f}s; startup elapsed: {time.perf_counter()-STARTED:.1f}s", flush=True)

    @Slot(str)
    def show_error(self, text: str):
        message = "Could not load Train 3 data\n\n" + text
        self.x_message.setText(message)
        self.y_message.setText(message)
        self.x_message.setPos(0.5, 0.5)
        self.y_message.setPos(0.5, 0.5)
        self.statusBar().showMessage(text)
        print(message, file=sys.stderr, flush=True)

    def worker_finished(self):
        if self.close_pending:
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.close_pending = True
            self.setWindowTitle("Finishing current database operation before closing…")
            event.ignore()
            return
        if self.thread is not None:
            self.thread.wait()
        event.accept()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH,
                        help="Database JSON file (default: beside script)")
    parser.add_argument("--cache-file", type=Path, default=DEFAULT_CACHE,
                        help="Local rendered-image cache path")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore local cache and re-query PostgreSQL")
    parser.add_argument("--no-cache", action="store_true",
                        help="Do not save a cache after a fresh database query")
    parser.add_argument("--buckets", type=int, default=RASTER_WIDTH,
                        help=f"Number of min/max buckets (default: {RASTER_WIDTH}; max: {RASTER_WIDTH})")
    parser.add_argument("--check-gui", action="store_true",
                        help="Test Qt window creation without connecting to PostgreSQL")
    args = parser.parse_args()
    if not 1 <= args.buckets <= RASTER_WIDTH:
        parser.error(f"--buckets must be between 1 and {RASTER_WIDTH}")

    print(f"Script: {Path(__file__).resolve()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"Qt binding: PySide6; pyqtgraph: {pg.__version__}", flush=True)
    print(f"Configuration: {args.config.resolve()}", flush=True)
    app = QApplication.instance() or QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)

    settings = None
    config_error = None
    if not args.check_gui:
        try:
            settings = read_settings(args.config)
        except Exception as exc:
            config_error = f"Configuration error: {exc}"

    use_cache = bool(args.cache_file.exists() and not args.refresh and not args.check_gui)
    if use_cache:
        print(f"Using local cache: {args.cache_file.resolve()} (use --refresh for fresh DB data)", flush=True)
    elif not args.check_gui:
        print("No usable cache selected; querying PostgreSQL for a fresh envelope.", flush=True)

    window = MainWindow(settings, config_error, args, use_cache)
    window.show()
    app.processEvents()
    print(f"Window shown in {time.perf_counter() - STARTED:.2f}s", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
