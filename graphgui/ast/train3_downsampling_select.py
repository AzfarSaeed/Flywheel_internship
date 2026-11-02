#!/usr/bin/env python3
"""Fast Train 3 X_g / Y_g viewer using PostgreSQL-side min/max downsampling.

The terminal asks what percentage of valid Train 3 measurements to remove.
PostgreSQL groups measurements into min/max buckets based on the requested
retained percentage. The actual retained/removed counts are reported for X_g
and Y_g separately. Min/max envelopes preserve bucket peaks and troughs.

Run (asks for downsampling percentage, e.g. 80.133):
    python dstrain_fast_optimized.py
Run with a preset percentage:
    python dstrain_fast_optimized.py --downsample-percent 80.133
Refresh from database instead of using a matching local cache:
    python dstrain_fast_optimized.py --refresh
Force a GUI-only test:
    python dstrain_fast_optimized.py --check-gui

The first database refresh still scans the matching rows on PostgreSQL and
its duration depends on server load, query plan, and network availability.
The saved local cache makes subsequent launches independent of the database.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
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
CACHE_VERSION = 2
FETCH_SIZE = 50_000


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


def build_stats_query(settings: Settings):
    """Return total/valid counts and CTID bounds in one aggregate query."""
    from psycopg2 import sql

    relation = sql.SQL("{}.{}").format(
        sql.Identifier(settings.schema), sql.Identifier(settings.table)
    )
    train = sql.Identifier(settings.train_column)
    x_col = sql.SQL("t.{}").format(sql.Identifier(settings.x_column))
    y_col = sql.SQL("t.{}").format(sql.Identifier(settings.y_column))
    x_expr = finite_value_expression(x_col)
    y_expr = finite_value_expression(y_col)
    block_expr = sql.SQL(
        "split_part(trim(both '()' from t.ctid::text), ',', 1)::bigint"
    )
    offset_expr = sql.SQL(
        "split_part(trim(both '()' from t.ctid::text), ',', 2)::bigint"
    )
    query = sql.SQL(
        "SELECT COUNT(*)::bigint, COUNT({x_expr})::bigint, COUNT({y_expr})::bigint, "
        "MIN({block_expr}), MAX({block_expr}), MAX({offset_expr}) "
        "FROM {relation} AS t WHERE t.{train} = %s"
    ).format(
        x_expr=x_expr, y_expr=y_expr, block_expr=block_expr,
        offset_expr=offset_expr, relation=relation, train=train
    )
    return query, (settings.train_value,)


def calculate_bucket_count(
    x_valid: int, y_valid: int, total: int, downsample_percent: float,
    page_span: int, max_offset: int, chronological: bool,
) -> int:
    """Choose enough min/max buckets to retain approximately the requested share.

    Two boundary values (min and max) are retained per bucket. At 0% requested
    downsampling, every row gets its own bucket so no source points are reduced.
    """
    if downsample_percent == 0:
        # In physical mode the CTID space contains one unique slot per possible
        # tuple location, so this also guarantees unique buckets for every row.
        return max(1, total if chronological else page_span * max_offset)

    keep_fraction = (100.0 - downsample_percent) / 100.0
    target_keep = int(math.ceil(max(x_valid, y_valid) * keep_fraction))
    target_buckets = max(1, int(math.ceil(target_keep / 2.0)))
    if not chronological:
        # The CTID position space includes unused slots; dividing it into
        # target_buckets still gives approximately total/target_buckets source
        # rows per bucket when tuple positions are reasonably distributed.
        target_buckets = min(target_buckets, max(1, page_span * max_offset))
    return max(1, target_buckets)


def build_bucket_query(
    settings: Settings, bucket_count: int, min_block: int,
    page_span: int, max_offset: int,
):
    """Build SQL that groups rows into the requested number of min/max buckets."""
    from psycopg2 import sql

    relation = sql.SQL("{}.{}").format(
        sql.Identifier(settings.schema), sql.Identifier(settings.table)
    )
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
            " SELECT (NTILE(%s) OVER (ORDER BY {order_expr}) - 1)::bigint AS bucket,"
            " {x_expr} AS xv, {y_expr} AS yv"
            " FROM {relation} AS t WHERE t.{train} = %s"
            ") "
            "SELECT bucket, COUNT(*)::bigint AS source_rows,"
            " COUNT(xv)::bigint AS x_valid, MIN(xv), MAX(xv),"
            " COUNT(yv)::bigint AS y_valid, MIN(yv), MAX(yv)"
            " FROM bucketed GROUP BY bucket"
        ).format(
            order_expr=order_expr, x_expr=x_expr, y_expr=y_expr,
            relation=relation, train=train
        )
        return query, (bucket_count, settings.train_value), "chronological"

    # Fast mode: map the physical tuple location (page, line-pointer offset)
    # into the requested bucket range. Unlike page-only grouping, including the
    # tuple offset permits more buckets than there are heap pages.
    block_expr = sql.SQL(
        "split_part(trim(both '()' from t.ctid::text), ',', 1)::bigint"
    )
    offset_expr = sql.SQL(
        "split_part(trim(both '()' from t.ctid::text), ',', 2)::bigint"
    )
    tuple_position = sql.SQL(
        "(({block} - %s)::numeric * %s::numeric + ({offset} - 1)::numeric)"
    ).format(block=block_expr, offset=offset_expr)
    bucket_expr = sql.SQL(
        "GREATEST(0::bigint, LEAST(%s::bigint - 1, "
        "FLOOR(({position} * %s::numeric) / (%s::numeric * %s::numeric))::bigint))"
    ).format(position=tuple_position)
    query = sql.SQL(
        "WITH bucketed AS ("
        " SELECT {bucket_expr} AS bucket, {x_expr} AS xv, {y_expr} AS yv"
        " FROM {relation} AS t WHERE t.{train} = %s"
        ") "
        "SELECT bucket, COUNT(*)::bigint AS source_rows,"
        " COUNT(xv)::bigint AS x_valid, MIN(xv), MAX(xv),"
        " COUNT(yv)::bigint AS y_valid, MIN(yv), MAX(yv)"
        " FROM bucketed GROUP BY bucket"
    ).format(
        bucket_expr=bucket_expr, x_expr=x_expr, y_expr=y_expr,
        relation=relation, train=train
    )
    params = (
        bucket_count, min_block, max_offset,
        bucket_count, page_span, max_offset, settings.train_value,
    )
    return query, params, "physical-ctid"


def _image_from_column_bounds(
    pixel_mins: np.ndarray, pixel_maxs: np.ndarray,
    height: int = RASTER_HEIGHT, width: int = RASTER_WIDTH,
):
    """Turn per-screen-column min/max bounds into an opaque graph image."""
    valid = np.isfinite(pixel_mins) & np.isfinite(pixel_maxs)
    if not valid.any():
        return np.zeros((height, width, 4), dtype=np.uint8), (-1.0, 1.0)

    valid_columns = np.flatnonzero(valid)
    if len(valid_columns) < width:
        # Physical CTID buckets can be empty when the table has deleted rows
        # or interleaved train records. Interpolate the envelope across display
        # columns so the plot remains continuous, without changing retained counts.
        all_columns = np.arange(width)
        pixel_mins = np.interp(all_columns, valid_columns, pixel_mins[valid_columns])
        pixel_maxs = np.interp(all_columns, valid_columns, pixel_maxs[valid_columns])
        valid = np.ones(width, dtype=bool)

    low_value = float(np.min(pixel_mins[valid]))
    high_value = float(np.max(pixel_maxs[valid]))
    y_range = finite_range(low_value, high_value)
    low, high = y_range
    span = high - low
    mask = np.zeros((height, width), dtype=bool)

    # Only the display columns are iterated here; database bucket rows have
    # already been folded into these columns using vectorized NumPy operations.
    for col in np.flatnonzero(valid):
        top = int(round((high - float(pixel_maxs[col])) / span * (height - 1)))
        bottom = int(round((high - float(pixel_mins[col])) / span * (height - 1)))
        top = max(0, min(height - 1, top))
        bottom = max(0, min(height - 1, bottom))
        if bottom < top:
            top, bottom = bottom, top
        mask[top:bottom + 1, col] = True

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask, 0:3] = 255
    rgba[mask, 3] = 255
    return np.ascontiguousarray(np.flipud(rgba)), y_range


def _accumulate_bucket_batch(
    rows, bucket_count: int,
    x_pixel_min: np.ndarray, x_pixel_max: np.ndarray,
    y_pixel_min: np.ndarray, y_pixel_max: np.ndarray,
) -> tuple[int, int, int, int, int, int]:
    """Fold a batch of SQL bucket summaries into fixed-size screen arrays."""
    bucket_ids = np.fromiter((int(r[0]) for r in rows), dtype=np.int64, count=len(rows))

    x_mins = np.fromiter(
        (np.nan if r[3] is None else float(r[3]) for r in rows),
        dtype=np.float64, count=len(rows)
    )
    x_maxs = np.fromiter(
        (np.nan if r[4] is None else float(r[4]) for r in rows),
        dtype=np.float64, count=len(rows)
    )
    y_mins = np.fromiter(
        (np.nan if r[6] is None else float(r[6]) for r in rows),
        dtype=np.float64, count=len(rows)
    )
    y_maxs = np.fromiter(
        (np.nan if r[7] is None else float(r[7]) for r in rows),
        dtype=np.float64, count=len(rows)
    )

    x_ok = np.isfinite(x_mins) & np.isfinite(x_maxs)
    y_ok = np.isfinite(y_mins) & np.isfinite(y_maxs)
    if bucket_count <= RASTER_WIDTH:
        # Spread each low-resolution bucket over its complete pixel interval,
        # matching the continuous envelope style of the previous viewer.
        for i, bucket_id in enumerate(bucket_ids):
            col_start = int(bucket_id * RASTER_WIDTH // bucket_count)
            col_end = int((bucket_id + 1) * RASTER_WIDTH // bucket_count)
            col_end = min(RASTER_WIDTH, max(col_start + 1, col_end))
            if x_ok[i]:
                x_pixel_min[col_start:col_end] = np.minimum(
                    x_pixel_min[col_start:col_end], x_mins[i]
                )
                x_pixel_max[col_start:col_end] = np.maximum(
                    x_pixel_max[col_start:col_end], x_maxs[i]
                )
            if y_ok[i]:
                y_pixel_min[col_start:col_end] = np.minimum(
                    y_pixel_min[col_start:col_end], y_mins[i]
                )
                y_pixel_max[col_start:col_end] = np.maximum(
                    y_pixel_max[col_start:col_end], y_maxs[i]
                )
    else:
        pixel_ids = (bucket_ids * RASTER_WIDTH // bucket_count).clip(0, RASTER_WIDTH - 1)
        if x_ok.any():
            np.minimum.at(x_pixel_min, pixel_ids[x_ok], x_mins[x_ok])
            np.maximum.at(x_pixel_max, pixel_ids[x_ok], x_maxs[x_ok])
        if y_ok.any():
            np.minimum.at(y_pixel_min, pixel_ids[y_ok], y_mins[y_ok])
            np.maximum.at(y_pixel_max, pixel_ids[y_ok], y_maxs[y_ok])

    x_kept = int(np.count_nonzero(x_ok) + np.count_nonzero(x_ok & (x_mins != x_maxs)))
    y_kept = int(np.count_nonzero(y_ok) + np.count_nonzero(y_ok & (y_mins != y_maxs)))
    source_rows = sum(int(r[1]) for r in rows)
    x_valid = sum(int(r[2]) for r in rows)
    y_valid = sum(int(r[5]) for r in rows)
    return source_rows, x_valid, y_valid, x_kept, y_kept, len(rows)


def parse_downsample_percent(raw: str) -> float:
    """Accept decimal percentage values such as 80, 80.133, or 35.598."""
    text = str(raw).strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        value = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError("Enter a number such as 80, 80.133, or 35.598.") from exc
    if not value.is_finite() or value < 0 or value >= 100:
        raise ValueError("Downsampling must be at least 0 and less than 100 percent.")
    return float(value)


def prompt_downsample_percent() -> float:
    """Prompt in the terminal; the entered value means percent to remove."""
    print("\nTrain 3 graph downsampling", flush=True)
    print("Enter the percentage of valid readings to REMOVE (0 to <100).", flush=True)
    print("Examples: 80  |  80.133  |  35.598", flush=True)
    print("The remaining percentage is represented by each bucket's min/max.\n", flush=True)
    while True:
        try:
            value = input("Downsample percentage: ")
        except EOFError as exc:
            raise RuntimeError(
                "No interactive terminal is available. Run from a terminal or pass --downsample-percent 80.133."
            ) from exc
        try:
            return parse_downsample_percent(value)
        except ValueError as exc:
            print(f"Invalid percentage: {exc}", flush=True)


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
        downsample_percent=np.float64(result["downsample_percent"]),
        bucket_count=np.int64(result["bucket_count"]),
        captured=np.asarray(captured), version=np.int64(CACHE_VERSION),
    )
    os.replace(temp_path, path)


def load_cache(path: Path, requested_percent: float) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        if int(z["version"]) != CACHE_VERSION:
            raise ValueError("The saved graph cache uses an older format; refresh is required.")
        cached_percent = float(z["downsample_percent"])
        if not math.isclose(cached_percent, requested_percent, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Cache was created with {cached_percent:g}% downsampling, "
                f"but you selected {requested_percent:g}%. A fresh database query is needed."
            )
        result = {
            "x_image": z["x_image"], "y_image": z["y_image"],
            "x_range": tuple(float(v) for v in z["x_range"]),
            "y_range": tuple(float(v) for v in z["y_range"]),
            "total": int(z["total"]), "x_valid": int(z["x_valid"]),
            "y_valid": int(z["y_valid"]), "x_retained": int(z["x_retained"]),
            "y_retained": int(z["y_retained"]),
            "downsample_percent": cached_percent,
            "bucket_count": int(z["bucket_count"]),
            "captured": str(z["captured"]),
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
                    result = load_cache(self.args.cache_file, self.args.downsample_percent)
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
        downsample_percent = float(self.args.downsample_percent)
        self.progress.emit("Connecting to PostgreSQL…")
        t0 = time.perf_counter()
        connection = connect(**s.connection)
        try:
            # Use a consistent read snapshot for the stats and aggregation queries.
            connection.set_session(isolation_level="REPEATABLE READ", readonly=True)
            stats_query, stats_params = build_stats_query(s)
            self.progress.emit("Counting Train 3 rows and sizing the requested buckets…")
            with connection.cursor() as cursor:
                cursor.execute(stats_query, stats_params)
                total, x_valid, y_valid, min_block, max_block, max_offset = cursor.fetchone()

            total = int(total or 0)
            x_valid = int(x_valid or 0)
            y_valid = int(y_valid or 0)
            if total == 0:
                raise RuntimeError(f"No rows found where {s.train_column} = {s.train_value!r}.")
            if x_valid == 0 and y_valid == 0:
                raise RuntimeError("Train 3 rows exist, but both X_g and Y_g contain no finite numeric values.")

            if min_block is None or max_block is None or max_offset is None:
                raise RuntimeError("Could not determine PostgreSQL tuple positions for Train 3.")
            page_span = max(1, int(max_block) - int(min_block) + 1)
            max_offset = max(1, int(max_offset))
            bucket_count = calculate_bucket_count(
                x_valid, y_valid, total, downsample_percent,
                page_span, max_offset, bool(s.order_column)
            )
            query, params, method = build_bucket_query(
                s, bucket_count, int(min_block), page_span, max_offset
            )
            self.progress.emit(
                f"Aggregating all {total:,} source rows into approximately "
                f"{bucket_count:,} min/max buckets ({downsample_percent:g}% requested downsampling)…"
            )
            if bucket_count > 100_000:
                print(
                    f"Note: {bucket_count:,} buckets are requested. A lower downsampling "
                    "percentage retains more detail but can take longer to aggregate and transfer.",
                    flush=True,
                )

            x_pixel_min = np.full(RASTER_WIDTH, np.inf, dtype=np.float64)
            x_pixel_max = np.full(RASTER_WIDTH, -np.inf, dtype=np.float64)
            y_pixel_min = np.full(RASTER_WIDTH, np.inf, dtype=np.float64)
            y_pixel_max = np.full(RASTER_WIDTH, -np.inf, dtype=np.float64)
            totals = {"source": 0, "x_valid": 0, "y_valid": 0,
                      "x_retained": 0, "y_retained": 0, "bucket_rows": 0}

            # Stream aggregate bucket rows to avoid fetchall() on a large result.
            with connection.cursor(name="train3_percent_bucket_stream") as cursor:
                cursor.itersize = FETCH_SIZE
                cursor.execute(query, params)
                while True:
                    rows = cursor.fetchmany(FETCH_SIZE)
                    if not rows:
                        break
                    batch = _accumulate_bucket_batch(
                        rows, bucket_count, x_pixel_min, x_pixel_max,
                        y_pixel_min, y_pixel_max
                    )
                    keys = ("source", "x_valid", "y_valid", "x_retained", "y_retained", "bucket_rows")
                    for key, value in zip(keys, batch):
                        totals[key] += value
                    if totals["bucket_rows"] % (FETCH_SIZE * 2) < FETCH_SIZE:
                        pct = min(100.0, totals["source"] * 100.0 / total)
                        self.progress.emit(
                            f"Aggregated {totals['source']:,}/{total:,} source rows ({pct:.1f}%)…"
                        )

        finally:
            connection.close()

        query_elapsed = time.perf_counter() - t0
        if totals["source"] != total:
            raise RuntimeError(
                f"The source row count changed or the aggregate was incomplete: "
                f"expected {total:,}, aggregated {totals['source']:,}."
            )
        self.progress.emit("Rendering X_g and Y_g graph images…")
        x_image, x_range = _image_from_column_bounds(x_pixel_min, x_pixel_max)
        y_image, y_range = _image_from_column_bounds(y_pixel_min, y_pixel_max)
        if totals["x_valid"] != x_valid or totals["y_valid"] != y_valid:
            raise RuntimeError(
                "The number of valid X_g/Y_g readings differed between the statistics "
                "and aggregation queries. Please refresh against a stable table."
            )
        return {
            "x_image": x_image, "y_image": y_image,
            "x_range": x_range, "y_range": y_range,
            "total": total, "x_valid": x_valid, "y_valid": y_valid,
            "x_retained": totals["x_retained"], "y_retained": totals["y_retained"],
            "bucket_rows": totals["bucket_rows"], "bucket_count": bucket_count,
            "downsample_percent": downsample_percent,
            "method": method, "query_elapsed": query_elapsed, "source": "DATABASE",
        }



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
        requested = float(result.get("downsample_percent", 0.0))
        status = (
            f"{source}: {total:,} source rows | requested removal {requested:g}% | "
            f"X_g kept {result['x_retained']:,}, removed {x_removed:,} ({x_pct:.3f}%) | "
            f"Y_g kept {result['y_retained']:,}, removed {y_removed:,} ({y_pct:.3f}%)"
        )
        if result.get("bucket_count") is not None:
            status += f" | buckets: {result['bucket_count']:,}"
        if result.get("method"):
            status += f" | grouping: {result['method']}"
        if result.get("query_elapsed") is not None:
            status += f" | database query: {result['query_elapsed']:.1f}s"
        if result.get("captured"):
            status += f" | cache saved: {result['captured']}"
        self.set_status(status)
        print("\nTrain 3 plot summary", flush=True)
        print(f"Source rows: {total:,}", flush=True)
        print(f"Requested downsampling: {requested:g}% (percentage of valid readings to remove)", flush=True)
        print(f"Min/max buckets requested: {result.get('bucket_count', 0):,}", flush=True)
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
    parser.add_argument("--downsample-percent", type=str,
                        help="Percentage of valid readings to remove (0 to <100); prompts if omitted")
    parser.add_argument("--check-gui", action="store_true",
                        help="Test Qt window creation without connecting to PostgreSQL")
    args = parser.parse_args()

    if args.check_gui:
        args.downsample_percent = 0.0
    elif args.downsample_percent is None:
        try:
            args.downsample_percent = prompt_downsample_percent()
        except (RuntimeError, KeyboardInterrupt) as exc:
            print(f"Cannot continue: {exc}", file=sys.stderr)
            return 2
    else:
        try:
            args.downsample_percent = parse_downsample_percent(args.downsample_percent)
        except ValueError as exc:
            parser.error(str(exc))

    print(f"Script: {Path(__file__).resolve()}", flush=True)
    print(f"Python: {sys.executable} ({sys.version.split()[0]})", flush=True)
    print(f"Qt binding: PySide6; pyqtgraph: {pg.__version__}", flush=True)
    print(f"Configuration: {args.config.resolve()}", flush=True)
    print(f"Requested data removal: {args.downsample_percent:g}%", flush=True)
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
        print(
            f"Checking local cache for {args.downsample_percent:g}% downsampling: "
            f"{args.cache_file.resolve()} (use --refresh for fresh DB data)", flush=True
        )
    elif not args.check_gui:
        print("No matching cache selected; querying PostgreSQL for a fresh envelope.", flush=True)

    window = MainWindow(settings, config_error, args, use_cache)
    window.show()
    app.processEvents()
    print(f"Window shown in {time.perf_counter() - STARTED:.2f}s", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
