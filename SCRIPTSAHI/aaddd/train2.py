#!/usr/bin/env python3
"""Fill metadata for the already-uploaded new_data.xlsx rows.

Keep this script, new_data.xlsx, and db_config.ini in the same folder.
Preview: python3 fill_metadata_train2.py --dry-run
Apply:   python3 fill_metadata_train2.py

Copies the original latitude/longitude sequence cyclically. Continues timestamps
at 8,000 samples/second, and sets the new rows' train_no to 2.
Does not insert rows or change the existing X/Y/Z or populated metadata.
"""

import argparse
import configparser
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta
import getpass
import math
import os
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent
SAMPLE_US = 125  # Original recording: 8,000 samples per second.


def display_time(value):
    return value.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def sample_time(start, index):
    """Millisecond precision, without accumulated rounding errors."""
    return start + timedelta(milliseconds=index * SAMPLE_US // 1000)


def read_numeric_columns(path, names, sheet_name=None):
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet_name is not None and sheet_name not in book.sheetnames:
            raise ValueError(f"Sheet {sheet_name!r} not found. Available: {book.sheetnames}")
        sheet = book[sheet_name] if sheet_name else book.worksheets[0]
        rows = sheet.iter_rows()
        headers = [str(c.value).strip() if c.value is not None else ""
                   for c in next(rows, ())]
        for name in names:
            if headers.count(name) != 1:
                raise ValueError(f"Excel row 1 must contain exactly one {name} header.")
        positions = [headers.index(name) for name in names]
        for number, row in enumerate(rows, 2):
            if all(c.value is None or (isinstance(c.value, str) and not c.value.strip()) for c in row):
                continue
            values = []
            for name, position in zip(names, positions):
                cell = row[position]
                try:
                    if cell.data_type in ("f", "e") or isinstance(cell.value, bool):
                        raise ValueError("use numeric values, not formulas/errors/booleans")
                    value = float(cell.value)
                    if not math.isfinite(value):
                        raise ValueError("number must be finite")
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f"Excel row {number}, {name}: {exc}") from exc
                values.append(value)
            yield tuple(values)
    finally:
        book.close()


def read_xyz(path, sheet_name=None):
    yield from read_numeric_columns(path, ("X_g", "Y_g", "Z_g"), sheet_name)


class SourceLocationOrderError(ValueError):
    """The original Excel order is needed to disambiguate locations."""


def prepare_source(records, ordered_locations=None):
    """Validate the original recording before calculating its continuation."""
    if not records:
        raise ValueError("No original rows with populated timestamp/latitude/longitude were found.")
    parsed = []
    for timestamp, latitude, longitude in records:
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(str(timestamp).strip())
        parsed.append((timestamp, latitude, longitude))
    parsed.sort(key=lambda row: row[0])
    start = parsed[0][0]
    if start.microsecond % 1000:
        raise ValueError("Original timestamps do not use the expected millisecond precision.")
    for index, row in enumerate(parsed):
        if row[0] != sample_time(start, index):
            raise ValueError("Original timestamps are not one continuous 8,000-sample/second recording. No changes made.")

    if ordered_locations is not None:
        if len(ordered_locations) != len(parsed):
            raise ValueError("The original Excel row count does not match the original database recording.")
        expected = Counter((sample_time(start, i), latitude, longitude)
                           for i, (latitude, longitude) in enumerate(ordered_locations))
        if expected != Counter(parsed):
            raise ValueError("The original Excel coordinates do not match the stored coordinates within each timestamp. No changes made.")
        return start, ordered_locations

    previous = None
    for row in parsed:
        if previous is not None and row[0] == previous[0] and row[1:] != previous[1:]:
            raise SourceLocationOrderError("The original Excel file is needed to recover coordinate order within shared timestamps.")
        previous = row
    return start, [(row[1], row[2]) for row in parsed]


def find_source_excel(filename=None):
    if filename:
        path = FOLDER / filename
        if path.is_file() and path.suffix.lower() == ".xlsx":
            return path
        raise ValueError(f"Original Excel file not found: {path}")
    # The original file is commonly alongside the script or one folder above.
    for path in (FOLDER / "data.xlsx", FOLDER.parent / "data.xlsx"):
        if path.is_file():
            return path
    raise ValueError("Place the original data.xlsx beside this script, or pass --source-excel /full/path/data.xlsx.")


def connect_settings(config_path):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(config_path, encoding="utf-8") or "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] settings in {config_path}")
    settings = config["postgresql"]
    connection = {key: settings.get(key, "") for key in ("host", "port", "dbname", "user")}
    if not all(str(value).strip() for value in connection.values()):
        raise ValueError("Set host, port, dbname, and user in db_config.ini.")
    connection["connect_timeout"] = 15
    password = os.environ.get("PGPASSWORD", settings.get("password"))
    if password is None:
        password = getpass.getpass("PostgreSQL password: ")
    if password:
        connection["password"] = password
    return connection, settings.get("schema", "public"), settings.get("table", "test_adxl")


def repair(args):
    import psycopg
    from psycopg import sql

    path = FOLDER / args.excel_file
    if not path.is_file() or path.suffix.lower() != ".xlsx":
        raise ValueError(f"Excel file not found or not .xlsx: {path}")
    connection, schema, name = connect_settings(FOLDER / args.config)
    table = sql.Identifier(schema, name)
    with psycopg.connect(**connection) as db:
        with db.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout = '15s'")
            # Prevent concurrent writes and row movement while temporary row
            # locations are used. Ordinary readers can still read the table.
            cur.execute(sql.SQL("LOCK TABLE {} IN EXCLUSIVE MODE").format(table))
            cur.execute(sql.SQL('''
                SELECT count(*),
                    count(*) FILTER (WHERE "timestamp" IS NULL AND latitude IS NULL AND longitude IS NULL),
                    count(*) FILTER (WHERE "timestamp" IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL)
                FROM {} WHERE train_no IS NULL
            ''').format(table))
            target_count, empty_count, complete_count = cur.fetchone()
            if not target_count:
                print("No new rows with NULL train_no found. Nothing changed.")
                return
            if empty_count != target_count and complete_count != target_count:
                raise ValueError("The new rows have a mixture of empty and populated metadata. Stopped without changing them; please inspect the table before continuing.")

            cur.execute(sql.SQL('''
                SELECT "timestamp", latitude, longitude FROM {}
                WHERE train_no = %s AND "timestamp" IS NOT NULL
                  AND latitude IS NOT NULL AND longitude IS NOT NULL
            ''').format(table), (args.source_train_no,))
            original_records = cur.fetchall()
            try:
                start, locations = prepare_source(original_records)
            except SourceLocationOrderError:
                original_path = find_source_excel(args.source_excel)
                print(f"Recovering coordinate order from: {original_path}")
                # Cast Excel coordinates through the actual database column types
                # before comparison, preserving the original import's rounding.
                cur.execute(sql.SQL('''
                    CREATE TEMP TABLE original_location_order ON COMMIT DROP AS
                    SELECT 0::bigint AS sample_index, latitude, longitude
                    FROM {} WITH NO DATA
                ''').format(table))
                with cur.copy('COPY original_location_order (sample_index,latitude,longitude) FROM STDIN') as copy:
                    with closing(read_numeric_columns(original_path, ("lat", "longi"), args.source_sheet)) as original_rows:
                        for index, location in enumerate(original_rows):
                            copy.write_row((index, *location))
                cur.execute('SELECT latitude,longitude FROM original_location_order ORDER BY sample_index')
                start, locations = prepare_source(original_records, cur.fetchall())
            source_count = len(locations)

            # Temporary tables copy the real column types, including any numeric
            # rounding, so XYZ comparison matches the earlier COPY import.
            cur.execute(sql.SQL('''
                CREATE TEMP TABLE metadata_stage ON COMMIT DROP AS
                SELECT 0::bigint AS sample_index, x_g, y_g, z_g,
                       "timestamp", latitude, longitude FROM {} WITH NO DATA
            ''').format(table))
            count = 0
            with cur.copy('COPY metadata_stage (sample_index,x_g,y_g,z_g,"timestamp",latitude,longitude) FROM STDIN') as copy:
                with closing(read_xyz(path, args.sheet)) as rows:
                    for index, xyz in enumerate(rows):
                        timestamp = sample_time(start, source_count + index)
                        latitude, longitude = locations[index % source_count]
                        # The naive timestamps used by this recording have exactly
                        # three fractional digits; preserve offsets if present.
                        timestamp_text = timestamp.isoformat(sep=" ", timespec="milliseconds")
                        copy.write_row((index, *xyz, timestamp_text, latitude, longitude))
                        count += 1
            if count != target_count:
                raise ValueError(f"Excel has {count:,} data rows, but {target_count:,} database rows need filling. No changes made.")

            cur.execute(sql.SQL('''
                CREATE TEMP TABLE metadata_targets ON COMMIT DROP AS
                SELECT tableoid AS row_table, ctid AS row_tid, x_g, y_g, z_g,
                       "timestamp" AS existing_timestamp,
                       latitude AS existing_latitude, longitude AS existing_longitude,
                       row_number() OVER (PARTITION BY x_g,y_g,z_g ORDER BY tableoid,ctid) AS occurrence
                FROM {} WHERE train_no IS NULL
            ''').format(table))
            cur.execute('''
                SELECT EXISTS (
                    (SELECT x_g,y_g,z_g FROM metadata_stage
                     EXCEPT ALL SELECT x_g,y_g,z_g FROM metadata_targets)
                    UNION ALL
                    (SELECT x_g,y_g,z_g FROM metadata_targets
                     EXCEPT ALL SELECT x_g,y_g,z_g FROM metadata_stage)
                )
            ''')
            if cur.fetchone()[0]:
                raise ValueError("Excel XYZ values do not exactly match the uploaded rows needing metadata. No changes made.")

            cur.execute('''
                CREATE TEMP TABLE metadata_mapping ON COMMIT DROP AS
                WITH ranked AS (
                    SELECT *, row_number() OVER (
                        PARTITION BY x_g,y_g,z_g ORDER BY sample_index) AS occurrence
                    FROM metadata_stage
                )
                SELECT t.row_table, t.row_tid, t.existing_timestamp,
                       t.existing_latitude, t.existing_longitude, s.* FROM ranked s
                JOIN metadata_targets t USING (x_g,y_g,z_g,occurrence)
            ''')
            cur.execute('SELECT count(*) FROM metadata_mapping')
            if cur.fetchone()[0] != count:
                raise ValueError("Unable to match every Excel sample to a database row.")
            print(f"Original rows: {source_count:,}; rows to fill: {count:,}")
            if empty_count:
                print(f"New timestamps: {display_time(sample_time(start, source_count))} to {display_time(sample_time(start, source_count + count - 1))}")
            else:
                print("All new rows already have metadata; only train_no will be changed.")
            if args.dry_run:
                db.rollback()
                print("Preview passed. No database rows changed.")
                return

            # ctids are only used inside this locked transaction, never as a
            # permanent identifier or as the sample order. Excel supplies order.
            cur.execute(sql.SQL('''
                WITH changed AS (
                    UPDATE {} AS d SET
                        "timestamp" = COALESCE(d."timestamp", m."timestamp"),
                        latitude = COALESCE(d.latitude, m.latitude),
                        longitude = COALESCE(d.longitude, m.longitude),
                        train_no = 2
                    FROM metadata_mapping m
                    WHERE d.tableoid = m.row_table AND d.ctid = m.row_tid
                      AND d.train_no IS NULL
                    RETURNING
                        d."timestamp" IS NOT DISTINCT FROM COALESCE(m.existing_timestamp, m."timestamp")
                        AND d.latitude IS NOT DISTINCT FROM COALESCE(m.existing_latitude, m.latitude)
                        AND d.longitude IS NOT DISTINCT FROM COALESCE(m.existing_longitude, m.longitude)
                        AND d.x_g IS NOT DISTINCT FROM m.x_g
                        AND d.y_g IS NOT DISTINCT FROM m.y_g
                        AND d.z_g IS NOT DISTINCT FROM m.z_g
                        AND d.train_no = 2 AS correct
                )
                SELECT count(*), count(*) FILTER (WHERE NOT correct) FROM changed
            ''').format(table))
            updated, incorrect = cur.fetchone()
            if updated != count or incorrect:
                raise ValueError("Database verification failed. The entire update will be rolled back.")
        # Exiting the connection commits only after all checks pass.
    print(f"Updated {updated:,} rows. train_no is 2; XYZ and original rows are unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel_file", nargs="?", default="new_data.xlsx")
    parser.add_argument("--config", default="db_config.ini")
    parser.add_argument("--sheet", help="Defaults to the first worksheet.")
    parser.add_argument("--source-train-no", type=int, default=1)
    parser.add_argument("--source-excel", help="Original data.xlsx; automatically found beside the script or one folder above when needed.")
    parser.add_argument("--source-sheet", help="Original workbook worksheet; defaults to its first worksheet.")
    parser.add_argument("--dry-run", action="store_true", help="Check matching and preview timestamps without updating rows.")
    args = parser.parse_args()
    try:
        repair(args)
    except ImportError:
        print(f"Install packages with: {sys.executable} -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Update failed: {exc}\nNo update was committed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
