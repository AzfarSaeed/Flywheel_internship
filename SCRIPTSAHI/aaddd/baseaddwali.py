#!/usr/bin/env python3
"""Append Excel X_g, Y_g, Z_g and train_no values to PostgreSQL.

Keep this script, the new .xlsx file, and your existing db_config.ini together.
Install: python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'
Run:     python3 append_xyz_train.py "new_data.xlsx"
Check:   python3 append_xyz_train.py "new_data.xlsx" --validate-only

No existing rows or columns are updated, deleted, or replaced. Other database
columns receive their defaults (or NULL when allowed). Running twice appends
the same data twice. The Excel file is read without modifying it.
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
EXCEL_COLUMNS = ("X_g", "Y_g", "Z_g", "train_no", "lat", "longi")
DB_COLUMNS = ("x_g", "y_g", "z_g", "train_no", "latitude", "longitude")


def read_xyz(path, sheet_name=None):
    from openpyxl import load_workbook

    if path.suffix.lower() != ".xlsx":
        raise ValueError("The input must be an .xlsx file.")
    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet_name is not None and sheet_name not in workbook.sheetnames:
            raise ValueError(f"Sheet {sheet_name!r} not found. Available: {workbook.sheetnames}")
        sheet = workbook[sheet_name] if sheet_name is not None else workbook.worksheets[0]
        rows = sheet.iter_rows()
        headers = [str(cell.value).strip() if cell.value is not None else ""
                   for cell in next(rows, ())]
        for name in EXCEL_COLUMNS:
            if headers.count(name) != 1:
                raise ValueError(f"Row 1 must contain exactly one {name!r} header.")
        positions = [headers.index(name) for name in EXCEL_COLUMNS]
        for row_number, row in enumerate(rows, start=2):
            if all(cell.value is None or
                   (isinstance(cell.value, str) and not cell.value.strip()) for cell in row):
                continue
            values = []
            for name, position in zip(EXCEL_COLUMNS, positions):
                cell = row[position]
                try:
                    if cell.data_type in ("f", "e"):
                        raise ValueError("replace Excel formulas/errors with numeric values")
                    if cell.value is None or isinstance(cell.value, bool):
                        raise ValueError("a numeric value is required")
                    if name == "train_no":
                        number = Decimal(str(cell.value).strip())
                        if not number.is_finite() or number != number.to_integral_value():
                            raise ValueError("train_no must be a whole number")
                        value = int(number)
                    else:
                        value = float(cell.value)
                        if not math.isfinite(value):
                            raise ValueError("value must be finite")
                except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
                    raise ValueError(f"Excel row {row_number}, {name}: {exc}") from exc
                values.append(value)
            yield tuple(values)
    finally:
        workbook.close()


def append_xyz(path, config_path, sheet_name=None):
    import psycopg
    from psycopg import sql

    config = configparser.ConfigParser(interpolation=None)
    if not config.read(config_path, encoding="utf-8") or "postgresql" not in config:
        raise ValueError(f"Missing [postgresql] settings in {config_path}")
    settings = config["postgresql"]
    connection = {}
    for key in ("host", "port", "dbname", "user"):
        if not settings.get(key, "").strip():
            raise ValueError(f"Set {key} in {config_path.name}")
        connection[key] = settings[key]
    connection["connect_timeout"] = 15
    password = os.environ.get("PGPASSWORD", settings.get("password"))
    if password is None:
        password = getpass.getpass("PostgreSQL password (Enter to use .pgpass): ")
    if password:
        connection["password"] = password

    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(settings.get("schema", "public"), settings.get("table", "test_adxl")),
        sql.SQL(", ").join(sql.Identifier(column) for column in DB_COLUMNS),
    )
    count = 0
    # COPY appends only these four fields. Other columns use database defaults.
    # The connection context commits only after complete success; errors roll back.
    with psycopg.connect(**connection) as database:
        with database.cursor() as cursor:
            with cursor.copy(statement) as copy:
                with closing(read_xyz(path, sheet_name)) as rows:
                    for values in rows:
                        copy.write_row(values)
                        count += 1
                if count == 0:
                    raise ValueError("The worksheet contains no data rows.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("excel_file", nargs="?", default="new_data.xlsx")
    parser.add_argument("--sheet", help="Worksheet name; defaults to the first worksheet.")
    parser.add_argument("--config", default="db_config.ini")
    parser.add_argument("--validate-only", action="store_true", help="Check Excel without connecting to the database.")
    args = parser.parse_args()
    path = FOLDER / args.excel_file
    try:
        if args.validate_only:
            count = sum(1 for _ in read_xyz(path, args.sheet))
            if not count:
                raise ValueError("The worksheet contains no data rows.")
            print(f"Validated {count:,} rows. No database changes made.")
        else:
            count = append_xyz(path, FOLDER / args.config, args.sheet)
            print(f"Successfully appended {count:,} rows. Previous rows were not replaced.")
    except ImportError:
        print(f"Install packages using this Python: {sys.executable} -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        print("The import was not committed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
