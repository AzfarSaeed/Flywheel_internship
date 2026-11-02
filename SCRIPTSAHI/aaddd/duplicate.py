#!/usr/bin/env python3
"""Fill only NULL timestamps for train_no=2 in PostgreSQL.

Generated timestamps start at 2026-10-12 07:28:59.369 and are spread across
10 seconds, ending at 2026-10-12 07:29:09.369 (millisecond format).

IMPORTANT: The UPDATE statement changes ONLY the "timestamp" column, and only
for rows where train_no=2 AND timestamp IS NULL. It does not insert/delete rows
or change any other column. Preview is the default; pass --apply to commit.

Requires: python3 -m pip install 'psycopg[binary]>=3.1,<4'
Preview:  python3 update_only_train2_null_timestamps_oct12.py
Apply:    python3 update_only_train2_null_timestamps_oct12.py --apply
"""

import argparse
import configparser
from datetime import datetime, timedelta
import getpass
import os
from pathlib import Path
import random
import sys

FOLDER = Path(__file__).resolve().parent
FIXED_START = datetime(2026, 10, 12, 7, 28, 59, 369000)
DEFAULT_DURATION_SECONDS = 10
DEFAULT_SEED = 42
DEFAULT_TRAIN_NO = 2


def read_config(config_path: Path):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(config_path, encoding="utf-8") or "postgresql" not in config:
        raise ValueError(f"Cannot read [postgresql] settings from {config_path}")
    settings = config["postgresql"]
    conninfo = {}
    for key in ("host", "port", "dbname", "user"):
        value = settings.get(key, "").strip()
        if not value:
            raise ValueError(f"Set {key!r} in {config_path.name}")
        conninfo[key] = value
    conninfo["connect_timeout"] = 15
    password = os.environ.get("PGPASSWORD", settings.get("password"))
    if password is None:
        password = getpass.getpass("PostgreSQL password (Enter to use .pgpass): ")
    if password:
        conninfo["password"] = password
    return conninfo, settings


def fmt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def random_offsets(count: int, duration_ms: int, seed: int):
    """Create sorted random millisecond offsets including both window endpoints."""
    if count <= 0:
        return []
    if duration_ms < 1:
        raise ValueError("Duration must be at least 1 millisecond.")
    if count == 1:
        return [0]
    if count == 2:
        return [0, duration_ms]
    rng = random.Random(seed)
    offsets = [0, duration_ms]
    offsets.extend(rng.randint(0, duration_ms) for _ in range(count - 2))
    offsets.sort()
    return offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(FOLDER / "db_config.ini"),
                        help="Path to db_config.ini (default: beside this script).")
    parser.add_argument("--schema", help="Override schema from config (default: public).")
    parser.add_argument("--table", help="Override table from config (default: test_adxl).")
    parser.add_argument("--train-no", type=int, default=DEFAULT_TRAIN_NO,
                        help="Train number whose NULL timestamps should be filled (default: 2).")
    parser.add_argument("--duration-seconds", type=int, default=DEFAULT_DURATION_SECONDS,
                        help="Timestamp window length in seconds (default: 10).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed; preview and apply produce the same sequence (default: 42).")
    parser.add_argument("--order-column", default=None,
                        help="Optional stable unique/order column; otherwise temporary ctid order is used.")
    parser.add_argument("--apply", action="store_true", help="Commit updates; without this, preview only.")
    args = parser.parse_args()

    try:
        import psycopg
        from psycopg import sql
    except ImportError:
        print("Missing dependency. Install with: python3 -m pip install 'psycopg[binary]>=3.1,<4'", file=sys.stderr)
        return 1

    try:
        if args.duration_seconds <= 0:
            raise ValueError("--duration-seconds must be positive.")
        duration_ms = args.duration_seconds * 1000
        conninfo, settings = read_config(Path(args.config).expanduser().resolve())
        schema = args.schema or settings.get("schema", "public").strip() or "public"
        table_name = args.table or settings.get("table", "test_adxl").strip() or "test_adxl"
        table = sql.Identifier(schema, table_name)
        ts_col = sql.Identifier("timestamp")
        train_col = sql.Identifier("train_no")

        with psycopg.connect(**conninfo) as conn:
            with conn.cursor() as cur:
                # Confirm that the two required columns exist and discover timestamp type.
                cur.execute(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s "
                    "AND column_name IN ('timestamp', 'train_no')",
                    (schema, table_name),
                )
                column_types = {name: data_type for name, data_type in cur.fetchall()}
                missing = {"timestamp", "train_no"} - set(column_types)
                if missing:
                    raise ValueError(f"Missing required database column(s): {', '.join(sorted(missing))}")

                timestamp_type = column_types["timestamp"]
                text_types = {"text", "character varying", "character"}
                timestamp_types = {"timestamp without time zone", "timestamp with time zone"}
                if timestamp_type not in text_types | timestamp_types:
                    raise ValueError(
                        f"Unsupported timestamp column type {timestamp_type!r}; expected text or a timestamp type."
                    )

                # On apply, block concurrent writers while selecting and updating the target rows,
                # so the temporary ctid mapping cannot be invalidated by a concurrent update.
                if args.apply:
                    cur.execute(sql.SQL("LOCK TABLE {} IN SHARE ROW EXCLUSIVE MODE").format(table))

                if args.order_column:
                    cur.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = %s AND table_name = %s AND column_name = %s",
                        (schema, table_name, args.order_column),
                    )
                    if cur.fetchone() is None:
                        raise ValueError(f"Order column {args.order_column!r} does not exist in {schema}.{table_name}.")
                    order_clause = sql.SQL("{} ASC, ctid ASC").format(sql.Identifier(args.order_column))
                else:
                    order_clause = sql.SQL("ctid ASC")

                # Only select target row locations; values in every other column are not read or altered.
                cur.execute(sql.SQL(
                    "SELECT ctid::text FROM {} WHERE {} = %s AND {} IS NULL ORDER BY {}"
                ).format(table, train_col, ts_col, order_clause), (args.train_no,))
                target_ctids = [row[0] for row in cur.fetchall()]
                if not target_ctids:
                    print(f"No NULL timestamps found for train_no={args.train_no}; nothing to update.")
                    return 0

                offsets = random_offsets(len(target_ctids), duration_ms, args.seed)
                if timestamp_type in text_types:
                    planned = [(ctid, fmt(FIXED_START + timedelta(milliseconds=offset)))
                               for ctid, offset in zip(target_ctids, offsets)]
                else:
                    planned = [(ctid, FIXED_START + timedelta(milliseconds=offset))
                               for ctid, offset in zip(target_ctids, offsets)]

                first_value = FIXED_START + timedelta(milliseconds=offsets[0])
                last_value = FIXED_START + timedelta(milliseconds=offsets[-1])
                print(f"Database: {conn.info.dbname}")
                print(f"Table: {schema}.{table_name}")
                print(f"Target train_no={args.train_no}: {len(target_ctids):,} NULL timestamps")
                print(f"Timestamp start: {fmt(first_value)}")
                print(f"Timestamp end:   {fmt(last_value)}")
                print(f"Timestamp format: YYYY-MM-DD HH:MM:SS.mmm; random seed: {args.seed}")
                print('ONLY the "timestamp" column will be updated; existing non-NULL timestamps and all other columns remain untouched.')
                if not args.order_column:
                    print("WARNING: assignment order uses PostgreSQL ctid (temporary physical row order).")

                if not args.apply:
                    print("\nPREVIEW ONLY — no database changes made. Re-run with --apply to commit these timestamp updates.")
                    return 0

                # Temporary patch holds only target ctid and replacement timestamp.
                # The sole UPDATE assignment below is target.timestamp = patch.new_timestamp.
                temp = sql.Identifier("timestamp_patch")
                cur.execute(sql.SQL(
                    "CREATE TEMP TABLE {} ON COMMIT DROP AS "
                    "SELECT ctid AS old_ctid, {} AS new_timestamp FROM {} WITH NO DATA"
                ).format(temp, ts_col, table))
                cur.execute(sql.SQL("CREATE UNIQUE INDEX ON {} (old_ctid)").format(temp))
                cur.executemany(
                    sql.SQL("INSERT INTO {} (old_ctid, new_timestamp) VALUES (%s::tid, %s)").format(temp),
                    planned,
                )
                cur.execute(sql.SQL(
                    "UPDATE {table} AS target SET {ts} = patch.new_timestamp "
                    "FROM {temp} AS patch WHERE target.ctid = patch.old_ctid "
                    "AND target.{ts} IS NULL AND target.{train} = %s"
                ).format(table=table, ts=ts_col, temp=temp, train=train_col), (args.train_no,))
                updated = cur.rowcount
                if updated != len(planned):
                    raise RuntimeError(
                        f"Expected to update {len(planned):,} rows but updated {updated:,}; transaction will roll back."
                    )
                print(f"\nSUCCESS: updated {updated:,} timestamp values. No other columns were changed.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("No changes were committed by this script.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
