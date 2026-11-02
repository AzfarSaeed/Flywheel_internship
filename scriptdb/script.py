#!/usr/bin/env python3
"""Append an Excel (.xlsx) or CSV file to PostgreSQL.
Keep this script, db_config.json, and your data file together.

Install: python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'
Check:   python3 upload_csv.py data.xlsx --check
Upload:  python3 upload_csv.py data.xlsx
Sheet:   python3 upload_csv.py data.xlsx --sheet 'Sheet1'
Relative paths are resolved from this script's folder.
"""

import argparse
import csv
import json
import sys
from contextlib import closing
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
# Time (s) is deliberately omitted: the pictured table has no time column.
COLUMN_MAP = {
    "X (g)": "x_g",
    "Y (g)": "y_g",
    "Z (g)": "z_g",
    "train_no": "train_no",
    "Timestamp": "timestamp",
    "lat": "latitude",
    "longi": "longitude",
}


def convert_value(column, value):
    if value is None:
        return None
    if column == "Timestamp":
        if isinstance(value, datetime):
            return value
        if isinstance(value, date):
            return datetime.combine(value, time.min)
        if not isinstance(value, str):
            raise ValueError(
                "Timestamp must be an Excel date/time cell or ISO date/time text, "
                "not an unformatted number or a time without a date."
            )
    value = str(value).strip()
    if not value:
        return None  # Empty cells become SQL NULL; zero remains zero.
    try:
        if column == "Timestamp":
            # ISO timestamp, e.g. 2026-10-26 22:00:00.000125 or ...+05:30.
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError("expected a finite number")
        if column == "train_no":
            if number != number.to_integral_value():
                raise ValueError("train_no must be a whole number")
            return int(number)
        return number
    except (ValueError, InvalidOperation) as exc:
        raise ValueError(f"Invalid {column} value {value!r}: {exc}") from exc


def csv_rows(path):
    """Stream and validate rows without loading the entire file into memory."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames
        if not headers:
            raise ValueError("The CSV is empty or has no header row.")
        headers = [name.strip() for name in headers]
        if len(headers) != len(set(headers)):
            raise ValueError("The CSV contains duplicate column names.")
        reader.fieldnames = headers
        missing = set(COLUMN_MAP) - set(headers)
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV line {reader.line_num}: wrong number of cells.")
            if all(not value.strip() for value in row.values()):
                continue
            try:
                yield tuple(convert_value(name, row[name]) for name in COLUMN_MAP)
            except ValueError as exc:
                raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc


def excel_rows(path, sheet_name=None):
    """Read .xlsx rows, preserving numeric cells and native Excel dates."""
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "Install Excel support first: python3 -m pip install 'openpyxl>=3.1,<4'"
        ) from exc

    # Inspect formulas explicitly so missing/stale cached results aren't uploaded.
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet_name is None:
            if not workbook.worksheets:
                raise ValueError("The workbook contains no worksheets.")
            sheet = workbook.worksheets[0]
        else:
            names = [item.title for item in workbook.worksheets]
            if sheet_name not in names:
                raise ValueError(
                    f"Worksheet {sheet_name!r} not found. Available: {', '.join(names)}"
                )
            sheet = workbook[sheet_name]
        # Some exporters report an incorrect used range. Read all stored rows.
        sheet.reset_dimensions()
        rows = sheet.iter_rows()
        first_row = next(rows, ())
        headers = [str(cell.value).strip() if cell.value is not None else ""
                   for cell in first_row]
        named_headers = [name for name in headers if name]
        if len(named_headers) != len(set(named_headers)):
            raise ValueError(f"Worksheet {sheet.title!r}: duplicate column names.")
        missing = set(COLUMN_MAP) - set(headers)
        if missing:
            raise ValueError(
                f"Worksheet {sheet.title!r}, row 1: missing columns: "
                + ", ".join(sorted(missing))
            )
        positions = [headers.index(name) for name in COLUMN_MAP]
        for row_number, cells in enumerate(rows, start=2):
            if all(cell.value is None or str(cell.value).strip() == "" for cell in cells):
                continue
            values = []
            for name, position in zip(COLUMN_MAP, positions):
                cell = cells[position] if position < len(cells) else None
                try:
                    if cell is not None and cell.data_type in {"f", "e"}:
                        raise ValueError(
                            "Contains a formula or Excel error. Resolve errors and "
                            "use Copy > Paste Special > Values for imported columns."
                        )
                    values.append(convert_value(name, cell.value if cell else None))
                except ValueError as exc:
                    raise ValueError(
                        f"Worksheet {sheet.title!r}, row {row_number}, {name}: {exc}"
                    ) from exc
            yield tuple(values)
    finally:
        workbook.close()


def data_rows(path, sheet_name=None):
    if path.suffix.lower() == ".xlsx":
        yield from excel_rows(path, sheet_name)
    elif path.suffix.lower() == ".csv":
        if sheet_name is not None:
            raise ValueError("--sheet is only available for Excel files.")
        yield from csv_rows(path)
    else:
        raise ValueError("Use an .xlsx or .csv file. Save older .xls files as .xlsx first.")


def upload(path, sheet_name=None):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError(
            "Install the database driver first: "
            "python3 -m pip install 'psycopg[binary]>=3.1,<4'"
        ) from exc

    config_path = FOLDER / "db_config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("db_config.json must contain a JSON object.")
    schema = config.pop("schema", "public")
    table = config.pop("table", "test_adxl")
    if not config.get("password"):
        raise ValueError(
            "Add your database password to the password field in "
            f"{config_path.name}. The script does not prompt for it."
        )
    config.setdefault("connect_timeout", 15)
    query = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(schema, table),
        sql.SQL(", ").join(sql.Identifier(name) for name in COLUMN_MAP.values()),
    )

    count = 0
    # A single transaction: any parsing/database error rolls back this upload.
    # Existing rows are kept. Running the upload again can create duplicates.
    try:
        with closing(data_rows(path, sheet_name)) as rows:
            with psycopg.connect(**config) as connection:
                with connection.cursor() as cursor:
                    with cursor.copy(query) as copy:
                        for values in rows:
                            copy.write_row(values)
                            count += 1
    except psycopg.OperationalError as exc:
        if "connection timeout expired" in str(exc).lower():
            raise RuntimeError(
                f"Connection timed out while reaching {config.get('host')}:{config.get('port')}. "
                "Check the host and port, connect to the required VPN, and ask the "
                "database administrator to allow your current network's IP address."
            ) from exc
        raise
    return count, schema, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_file", nargs="?", default="data.xlsx")
    parser.add_argument("--sheet", help="Excel worksheet name; defaults to the first worksheet.")
    parser.add_argument("--check", action="store_true",
                        help="Validate the file locally without connecting to PostgreSQL.")
    args = parser.parse_args()
    path = Path(args.data_file).expanduser()
    if not path.is_absolute():
        path = FOLDER / path
    try:
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        if path.suffix.lower() not in {".xlsx", ".csv"}:
            raise ValueError("Use an .xlsx or .csv file.")
        if args.sheet is not None and path.suffix.lower() != ".xlsx":
            raise ValueError("--sheet is only available for Excel files.")
        if args.check:
            count = sum(1 for _ in data_rows(path, args.sheet))
            print(f"File valid: {count:,} rows. No database connection was made.")
        else:
            count, schema, table = upload(path, args.sheet)
            print(f"Uploaded {count:,} rows to {schema}.{table}.")
    except KeyboardInterrupt:
        print("\nInterrupted. An uncommitted upload is rolled back.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
