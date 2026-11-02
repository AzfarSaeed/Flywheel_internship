#!/usr/bin/env python3
"""Append sensor Excel (.xlsx) rows to the existing PostgreSQL test_adxl table.

Keep this script, db_config.ini, and your Excel file in the same folder.
Requires Python 3.9+; install the database driver with:
    python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'
"""

import argparse
import configparser
from contextlib import closing
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
import getpass
import math
import os
from pathlib import Path
import re
import sys


FOLDER = Path(__file__).resolve().parent
DEFAULT_START = "2026-10-11 07:28:49.369"
COLUMN_MAP = {
    "X (g)": "x_g",
    "Y (g)": "y_g",
    "Z (g)": "z_g",
    "train_no": "train_no",
    "Timestamp": "timestamp",
    "lat": "latitude",
    "longi": "longitude",
}


def read_rows(excel_path, sheet_name=None):
    """Stream Excel rows; headers must be in row 1."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError("Install Excel support: python3 -m pip install 'openpyxl>=3.1,<4'") from exc
    if excel_path.suffix.lower() != ".xlsx":
        raise ValueError("Please supply an .xlsx Excel file.")
    # Read formulas explicitly so missing cached results cannot become NULL.
    workbook = load_workbook(excel_path, read_only=True, data_only=False)
    try:
        if sheet_name is not None and sheet_name not in workbook.sheetnames:
            raise ValueError(f"Unknown sheet {sheet_name!r}. Available: {', '.join(workbook.sheetnames)}")
        sheet = workbook[sheet_name] if sheet_name is not None else workbook.worksheets[0]
        rows = sheet.iter_rows()
        headers = [str(cell.value).strip() if cell.value is not None else "" for cell in next(rows, ())]
        named_headers = [name for name in headers if name]
        if len(named_headers) != len(set(named_headers)):
            raise ValueError("The worksheet contains duplicate column names.")
        missing = set(COLUMN_MAP) - set(named_headers)
        if missing:
            raise ValueError("Missing Excel columns in row 1: " + ", ".join(sorted(missing)))
        positions = {name: headers.index(name) for name in COLUMN_MAP}
        for line, row in enumerate(rows, start=2):
            if all(cell.value is None or (isinstance(cell.value, str) and not cell.value.strip()) for cell in row):
                continue
            values = []
            for column in COLUMN_MAP:
                if column == "Timestamp":
                    # Generate this value from row position during import.
                    values.append(None)
                    continue
                cell = row[positions[column]]
                value = cell.value.strip() if isinstance(cell.value, str) else cell.value
                if cell.data_type in ("f", "e"):
                    raise ValueError(f"Excel row {line}, {column}: replace formulas/errors with values before importing.")
                try:
                    if value is None or isinstance(value, bool):
                        raise ValueError("missing number or boolean")
                    if column == "train_no":
                        decimal = Decimal(str(value))
                        if not decimal.is_finite() or decimal != decimal.to_integral_value():
                            raise ValueError("train number must be an integer")
                        number = int(decimal)
                    else:
                        number = float(value)
                        if not math.isfinite(number):
                            raise ValueError("non-finite value")
                except (ValueError, TypeError, InvalidOperation, OverflowError) as exc:
                    raise ValueError(f"Excel row {line}: invalid {column} value {value!r}.") from exc
                values.append(number)
            yield tuple(values)
    finally:
        workbook.close()


def timestamp_at(start, index, row_count, duration_ms):
    """Floor each sample offset to milliseconds without floating-point drift."""
    offset_ms = index * duration_ms // row_count
    return (start + timedelta(milliseconds=offset_ms)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def timed_rows(excel_path, sheet_name=None, start_text=DEFAULT_START, duration_seconds="10"):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", start_text):
        raise ValueError("Start timestamp must use YYYY-MM-DD HH:MM:SS.mmm.")
    start = datetime.strptime(start_text, "%Y-%m-%d %H:%M:%S.%f")
    duration_ms = Decimal(str(duration_seconds)) * 1000
    if not duration_ms.is_finite() or duration_ms <= 0 or duration_ms != duration_ms.to_integral_value():
        raise ValueError("Duration must be positive and use at most millisecond precision.")
    duration_ms = int(duration_ms)
    # Count actual data rows, excluding blank rows, before calculating spacing.
    row_count = sum(1 for _ in read_rows(excel_path, sheet_name))
    if row_count == 0:
        raise ValueError("No data rows found.")
    timestamp_index = list(COLUMN_MAP).index("Timestamp")
    emitted = 0
    with closing(read_rows(excel_path, sheet_name)) as rows:
        for index, row in enumerate(rows):
            if index >= row_count:
                raise ValueError("Workbook changed while importing; close Excel and retry.")
            values = list(row)
            values[timestamp_index] = timestamp_at(start, index, row_count, duration_ms)
            yield tuple(values)
            emitted += 1
    if emitted != row_count:
        raise ValueError("Workbook changed while importing; close Excel and retry.")


def import_excel(excel_path, config_path, sheet_name=None, start_text=DEFAULT_START, duration_seconds="10"):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError(
            "Install the driver: python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'"
        ) from exc

    config = configparser.ConfigParser(interpolation=None)
    if not config.read(config_path, encoding="utf-8"):
        raise ValueError(f"Cannot read configuration: {config_path}")
    if "postgresql" not in config:
        raise ValueError("Configuration needs a [postgresql] section.")
    settings = config["postgresql"]
    for key in ("host", "port", "dbname", "user"):
        if not settings.get(key, "").strip():
            raise ValueError(f"Set {key} in {config_path.name}.")
    connection = {key: settings[key] for key in ("host", "port", "dbname", "user")}
    connection["connect_timeout"] = 15
    password = os.environ.get("PGPASSWORD", settings.get("password"))
    if password is None:
        password = getpass.getpass("PostgreSQL password (Enter to use .pgpass): ")
    if password:
        connection["password"] = password

    schema = settings.get("schema", "public")
    table = settings.get("table", "test_adxl")
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(schema, table),
        sql.SQL(", ").join(sql.Identifier(name) for name in COLUMN_MAP.values()),
    )
    count = 0
    # Commit only when the entire file succeeds; roll back on any error.
    with psycopg.connect(**connection) as database:
        with database.cursor() as cursor:
            with cursor.copy(statement) as copy:
                with closing(timed_rows(excel_path, sheet_name, start_text, duration_seconds)) as rows:
                    for values in rows:
                        copy.write_row(values)
                        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel_file", nargs="?", default="data.xlsx")
    parser.add_argument("--sheet", help="Worksheet name; defaults to the first worksheet.")
    parser.add_argument("--config", default="db_config.ini")
    parser.add_argument("--start-timestamp", default=DEFAULT_START)
    parser.add_argument("--duration-seconds", default="10", help="Recording duration; defaults to 10 seconds.")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Check Excel headers, cell types, and numeric values without connecting to the database.",
    )
    args = parser.parse_args()
    # Relative paths always resolve beside this script, regardless of where it runs.
    excel_path = FOLDER / args.excel_file
    try:
        if args.validate_only:
            count = 0
            first = last = None
            timestamp_index = list(COLUMN_MAP).index("Timestamp")
            for row in timed_rows(excel_path, args.sheet, args.start_timestamp, args.duration_seconds):
                last = row[timestamp_index]
                if first is None:
                    first = last
                count += 1
            print(f"Validated {count:,} rows. No database changes made.")
            print(f"Timestamp range: {first} to {last}")
        else:
            count = import_excel(excel_path, FOLDER / args.config, args.sheet, args.start_timestamp, args.duration_seconds)
            print(f"Successfully inserted {count:,} rows.")
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
