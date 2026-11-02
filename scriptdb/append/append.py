#!/usr/bin/env python3
"""Append X_g, Y_g, and Z_g values from an Excel file to PostgreSQL.

Place this script, db_config.json, and the .xlsx file in the same folder.

Install: python3 -m pip install 'psycopg[binary]>=3.1,<4' 'openpyxl>=3.1,<4'
Check:   python3 append_xyz_xlsx.py --check
Upload:  python3 append_xyz_xlsx.py

This script appends new rows. It never updates or deletes existing rows.
"""

import argparse
import json
import sys
from contextlib import closing
from decimal import Decimal, InvalidOperation
from pathlib import Path


FOLDER = Path(__file__).resolve().parent
DESTINATION_COLUMNS = ("x_g", "y_g", "z_g")


def normalise_header(value):
    """Treat X_g, x g, and X-g as the same source header."""
    return "".join("_" if character in " -" else character.lower() for character in str(value).strip())


def to_decimal(value, header, row_number):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"Row {row_number}, {header}: expected a number, found a boolean.")
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(
            f"Row {row_number}, {header}: expected a number, found {value!r}."
        ) from exc
    if not number.is_finite():
        raise ValueError(f"Row {row_number}, {header}: values must be finite numbers.")
    return number


def excel_rows(path, sheet_name=None):
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "Install Excel support first: "
            "python3 -m pip install 'openpyxl>=3.1,<4'"
        ) from exc

    workbook = load_workbook(path, read_only=True, data_only=False)
    try:
        if sheet_name is None:
            sheet = workbook.worksheets[0]
        elif sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
        else:
            raise ValueError(
                f"Worksheet {sheet_name!r} was not found. Available worksheets: "
                + ", ".join(workbook.sheetnames)
            )
        sheet.reset_dimensions()
        iterator = sheet.iter_rows()
        header_cells = next(iterator, ())
        headers = [normalise_header(cell.value) if cell.value is not None else ""
                   for cell in header_cells]
        positions = {}
        for source_name in DESTINATION_COLUMNS:
            matches = [index for index, header in enumerate(headers) if header == source_name]
            if len(matches) != 1:
                raise ValueError(
                    f"Worksheet {sheet.title!r}: expected exactly one {source_name!r} "
                    f"column in row 1; found {len(matches)}."
                )
            positions[source_name] = matches[0]
        for row_number, cells in enumerate(iterator, start=2):
            values = []
            for source_name in DESTINATION_COLUMNS:
                position = positions[source_name]
                cell = cells[position] if position < len(cells) else None
                if cell is not None and cell.data_type in {"f", "e"}:
                    raise ValueError(
                        f"Row {row_number}, {source_name}: formulas and Excel errors "
                        "are not accepted. Paste values before uploading."
                    )
                values.append(to_decimal(cell.value if cell else None, source_name, row_number))
            # Ignore blank Excel rows. Values containing a real zero are kept.
            if any(value is not None for value in values):
                yield tuple(values)
    finally:
        workbook.close()


def load_config():
    config_path = FOLDER / "db_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing {config_path.name} beside this script.") from exc
    if not isinstance(config, dict):
        raise ValueError("db_config.json must contain a JSON object.")
    schema = config.pop("schema", "public")
    table = config.pop("table", "test_adxl")
    if not config.get("password"):
        raise ValueError("Add the database password to db_config.json before running.")
    config.setdefault("connect_timeout", 15)
    return config, schema, table


def upload(path, sheet_name):
    try:
        import psycopg
        from psycopg import sql
    except ImportError as exc:
        raise RuntimeError(
            "Install the database driver first: "
            "python3 -m pip install 'psycopg[binary]>=3.1,<4'"
        ) from exc
    config, schema, table = load_config()
    statement = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier(schema, table),
        sql.SQL(", ").join(sql.Identifier(column) for column in DESTINATION_COLUMNS),
    )
    count = 0
    # One transaction: a bad row or database error rolls back this entire upload.
    with closing(excel_rows(path, sheet_name)) as rows:
        with psycopg.connect(**config) as connection:
            with connection.cursor() as cursor:
                with cursor.copy(statement) as copy:
                    for row in rows:
                        copy.write_row(row)
                        count += 1
    return count, schema, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "excel_file",
        nargs="?",
        default="newdata.xlsx",
        help="The .xlsx file containing X_g, Y_g, and Z_g (default: newdata.xlsx).",
    )
    parser.add_argument("--sheet", help="Worksheet name. Defaults to the first worksheet.")
    parser.add_argument("--check", action="store_true",
                        help="Validate the Excel file without connecting to PostgreSQL.")
    args = parser.parse_args()
    path = Path(args.excel_file).expanduser()
    if not path.is_absolute():
        path = FOLDER / path
    try:
        if path.suffix.lower() != ".xlsx":
            raise ValueError("Use an .xlsx Excel file.")
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        if args.check:
            count = sum(1 for _ in excel_rows(path, args.sheet))
            print(f"Excel file is valid: {count:,} rows. No database connection was made.")
        else:
            count, schema, table = upload(path, args.sheet)
            print(f"Appended {count:,} rows to {schema}.{table}.")
    except KeyboardInterrupt:
        print("\nInterrupted. Any uncommitted rows were rolled back.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
