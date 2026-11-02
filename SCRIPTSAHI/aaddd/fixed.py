#!/usr/bin/env python3
"""Append Excel X_g, Y_g, Z_g, train_no, lat, and longi to PostgreSQL.

Install:
    python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'

Run:
    python3 append_xyz_train_fixed.py "new_data.xlsx"
    python3 append_xyz_train_fixed.py "new_data.xlsx" --validate-only

Only the six listed target columns are inserted. Existing rows are not updated,
 deleted, or replaced. Blank cells in the incoming worksheet are forward-filled
from the most recent nonblank value in that same column. If a column starts with
blank cells and has no earlier value to copy, the script stops safely.

The import runs in a transaction: any error rolls back the entire import.
Running the same file twice will append its rows twice.
"""

import argparse
import configparser
from contextlib import closing
from decimal import Decimal, InvalidOperation
import getpass
import math
import os
from pathlib import Path
import sys

FOLDER = Path(__file__).resolve().parent

# Source headers in the Excel worksheet (must match exactly, ignoring spaces).
EXCEL_COLUMNS = ("X_g", "Y_g", "Z_g", "train_no", "lat", "longi")
# Destination columns in PostgreSQL.
DB_COLUMNS = ("x_g", "y_g", "z_g", "train_no", "latitude", "longitude")

MISSING_TEXT = {"", "nan", "null", "none", "na", "n/a"}


def is_missing(value):
    """Return True for empty Excel cells and common textual null markers."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in MISSING_TEXT:
        return True
    return False


def parse_numeric(value, column_name, row_number):
    """Parse a finite numeric value, keeping train_no as an integer."""
    if isinstance(value, bool):
        raise ValueError(
            f"Excel row {row_number}, {column_name}: Boolean is not a numeric value."
        )

    try:
        if column_name == "train_no":
            number = Decimal(str(value).strip())
            if not number.is_finite() or number != number.to_integral_value():
                raise ValueError("train_no must be a whole number")
            return int(number)

        number = float(value)
        if not math.isfinite(number):
            raise ValueError("value must be finite")
        return number
    except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
        raise ValueError(
            f"Excel row {row_number}, {column_name}: {exc}; found {value!r}"
        ) from exc


def read_xyz(path, sheet_name=None):
    """Yield six selected columns, forward-filling blank cells column by column."""
    from openpyxl import load_workbook

    path = Path(path)
    if path.suffix.lower() != ".xlsx":
        raise ValueError("The input must be an .xlsx file.")
    if not path.is_file():
        raise FileNotFoundError(f"Excel file not found: {path}")

    workbook = load_workbook(path, read_only=True, data_only=False)
    previous_values = {name: None for name in EXCEL_COLUMNS}
    try:
        if sheet_name is not None and sheet_name not in workbook.sheetnames:
            raise ValueError(
                f"Sheet {sheet_name!r} not found. Available sheets: {workbook.sheetnames}"
            )

        sheet = workbook[sheet_name] if sheet_name is not None else workbook.worksheets[0]
        rows = sheet.iter_rows()
        header_row = next(rows, None)
        if header_row is None:
            raise ValueError("The worksheet is empty.")

        headers = [str(cell.value).strip() if cell.value is not None else ""
                   for cell in header_row]
        for name in EXCEL_COLUMNS:
            if headers.count(name) != 1:
                raise ValueError(
                    f"Row 1 must contain exactly one {name!r} header; found {headers.count(name)}."
                )
        positions = [headers.index(name) for name in EXCEL_COLUMNS]

        count = 0
        for row_number, row in enumerate(rows, start=2):
            # Ignore entirely empty worksheet rows.
            if all(is_missing(cell.value) for cell in row):
                continue

            values = []
            for name, position in zip(EXCEL_COLUMNS, positions):
                cell = row[position]

                if cell.data_type == "f":
                    raise ValueError(
                        f"Excel row {row_number}, {name}: the cell contains a formula. "
                        "Replace it with a numeric value before importing."
                    )
                if cell.data_type == "e":
                    raise ValueError(
                        f"Excel row {row_number}, {name}: the cell contains an Excel error "
                        f"({cell.value!r}). Fix it before importing."
                    )

                raw_value = cell.value
                if is_missing(raw_value):
                    previous = previous_values[name]
                    if previous is None:
                        raise ValueError(
                            f"Excel row {row_number}, {name}: cell is blank and there is no "
                            "earlier nonblank value in this worksheet to copy. Fill this "
                            "cell manually or provide a valid starting value."
                        )
                    value = previous
                else:
                    value = parse_numeric(raw_value, name, row_number)
                    previous_values[name] = value

                values.append(value)

            count += 1
            yield tuple(values)

        if count == 0:
            raise ValueError("The worksheet contains no data rows.")
    finally:
        workbook.close()


def get_connection_settings(config_path):
    config = configparser.ConfigParser(interpolation=None)
    if not config.read(config_path, encoding="utf-8") or "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] settings in {config_path}")

    settings = config["postgresql"]
    connection = {}
    for key in ("host", "port", "dbname", "user"):
        if not settings.get(key, "").strip():
            raise ValueError(f"Set {key} in {config_path.name}")
        connection[key] = settings[key].strip()

    connection["connect_timeout"] = 15
    password = os.environ.get("PGPASSWORD", settings.get("password"))
    if password is None:
        password = getpass.getpass(
            "PostgreSQL password (press Enter to use your .pgpass file): "
        )
    if password:
        connection["password"] = password
    return connection, settings


def append_xyz(path, config_path, sheet_name=None):
    import psycopg
    from psycopg import sql

    connection, settings = get_connection_settings(config_path)
    schema_name = settings.get("schema", "public").strip() or "public"
    table_name = settings.get("table", "test_adxl").strip() or "test_adxl"

    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(schema_name, table_name),
        sql.SQL(", ").join(sql.Identifier(column) for column in DB_COLUMNS),
    )

    count = 0
    # The connection context commits only if the full COPY succeeds; otherwise it rolls back.
    with psycopg.connect(**connection) as database:
        with database.cursor() as cursor:
            with cursor.copy(statement) as copy:
                with closing(read_xyz(path, sheet_name)) as rows:
                    for values in rows:
                        copy.write_row(values)
                        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel_file", nargs="?", default="new_data.xlsx")
    parser.add_argument("--sheet", help="Worksheet name; defaults to the first worksheet.")
    parser.add_argument("--config", default="db_config.ini")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the Excel file without connecting to PostgreSQL.",
    )
    args = parser.parse_args()

    excel_path = Path(args.excel_file).expanduser()
    if not excel_path.is_absolute():
        excel_path = FOLDER / excel_path
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = FOLDER / config_path

    try:
        if args.validate_only:
            count = sum(1 for _ in read_xyz(excel_path, args.sheet))
            print(f"Validated {count:,} rows. No database changes made.")
        else:
            count = append_xyz(excel_path, config_path, args.sheet)
            print(
                f"Successfully appended {count:,} rows to the database. "
                "Existing rows and other columns were not updated."
            )
    except ImportError:
        print(
            f"Install packages using this Python: {sys.executable} -m pip install "
            "'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        print("The import was not committed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
