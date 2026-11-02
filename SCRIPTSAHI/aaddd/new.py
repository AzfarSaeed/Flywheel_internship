"""Append only X_g, Y_g, Z_g and train_no from newdata.xlsx to PostgreSQL.

Existing rows are never updated or deleted. Blank values in the Excel file are
forward-filled from the previous row; leading blanks are seeded from the latest
non-null value already stored in the database (ordered by timestamp).

Install dependencies:
    pip install pandas openpyxl psycopg2-binary
"""

import os
import sys
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


# CHANGE THIS to the full path of newdata.xlsx on your computer.
EXCEL_PATH = r"C:\Users\YOUR_USERNAME\Downloads\newdata.xlsx"

# Update these values to match your PostgreSQL connection details.
# Environment variables are supported; otherwise replace the placeholders.
DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "ofc_daqhat_test"),
    "user": os.getenv("PGUSER", "YOUR_DB_USERNAME"),
    "password": os.getenv("PGPASSWORD", "YOUR_DB_PASSWORD"),
}

SCHEMA = "public"
TABLE = "test_adxl"
ORDER_COLUMN = "timestamp"  # Present in the table shown in your screenshot.

# These are the ONLY columns inserted by this script.
SOURCE_TO_DB = {
    "x_g": "x_g",
    "y_g": "y_g",
    "z_g": "z_g",
    "train_no": "train_no",
}


def load_excel() -> pd.DataFrame:
    df = pd.read_excel(EXCEL_PATH, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]

    # Match column headers without being sensitive to capitalization.
    header_lookup = {str(c).strip().casefold(): c for c in df.columns}
    required = list(SOURCE_TO_DB.keys())
    missing = [name for name in required if name.casefold() not in header_lookup]
    if missing:
        raise ValueError(
            "These required columns were not found in newdata.xlsx: "
            + ", ".join(missing)
            + f"\nColumns found: {list(df.columns)}"
        )

    # Select by header, not by position, so columns cannot get mixed up.
    selected = [header_lookup[name.casefold()] for name in required]
    df = df[selected].copy()
    df.columns = required

    # Convert empty strings and non-numeric cells to missing values.
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df.empty:
        raise ValueError("newdata.xlsx contains no data rows.")

    return df


def append_data() -> None:
    df = load_excel()
    db_columns = list(SOURCE_TO_DB.values())

    conn = psycopg2.connect(**DB_CONFIG)
    inserted = 0
    try:
        # All schema checks and inserts are one transaction. Any error rolls it back.
        with conn:
            with conn.cursor() as cur:
                table_ident = sql.SQL("{}.{}").format(
                    sql.Identifier(SCHEMA), sql.Identifier(TABLE)
                )

                # Create target columns only if they do not already exist.
                # Existing columns and their values are not overwritten.
                cur.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS x_g DOUBLE PRECISION").format(table_ident)
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS y_g DOUBLE PRECISION").format(table_ident)
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS z_g DOUBLE PRECISION").format(table_ident)
                )
                cur.execute(
                    sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS train_no BIGINT").format(table_ident)
                )

                # Inspect schema and stop if another column must be supplied for each new row.
                cur.execute(
                    """
                    SELECT column_name, is_nullable, column_default, is_identity, is_generated
                    FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s
                    """,
                    (SCHEMA, TABLE),
                )
                schema_info = cur.fetchall()
                existing_columns = {row[0] for row in schema_info}
                if ORDER_COLUMN not in existing_columns:
                    raise RuntimeError(
                        f'The table has no "{ORDER_COLUMN}" column. Set ORDER_COLUMN '
                        "to a reliable existing ordering column before running this script."
                    )

                required_other_columns = [
                    name
                    for name, nullable, default, identity, generated in schema_info
                    if name not in db_columns
                    and nullable == "NO"
                    and default is None
                    and identity != "YES"
                    and generated != "ALWAYS"
                ]
                if required_other_columns:
                    raise RuntimeError(
                        "The database requires values for these additional NOT NULL columns: "
                        + ", ".join(required_other_columns)
                        + ". The Excel file shown does not provide them. No rows were inserted. "
                        "Do not remove these checks; supply legitimate values for those fields "
                        "or use a table where those fields have defaults/allow NULL."
                    )

                # Get the latest non-null previous value for each target column.
                # These values are used only to fill leading blanks in the new Excel data.
                previous_values = {}
                for col in db_columns:
                    cur.execute(
                        sql.SQL(
                            "SELECT {col} FROM {table} "
                            "WHERE {col} IS NOT NULL "
                            "ORDER BY {order_col} DESC NULLS LAST LIMIT 1"
                        ).format(
                            col=sql.Identifier(col),
                            table=table_ident,
                            order_col=sql.Identifier(ORDER_COLUMN),
                        )
                    )
                    found = cur.fetchone()
                    previous_values[col] = found[0] if found else None

                # Fill blanks only in the incoming file, never in existing DB rows.
                # Interior/trailing blanks use the previous nonblank row in this file;
                # leading blanks use the previous database value where available.
                for col in db_columns:
                    df[col] = df[col].ffill()
                    if previous_values[col] is not None:
                        df[col] = df[col].fillna(previous_values[col])
                    if df[col].isna().any():
                        count = int(df[col].isna().sum())
                        raise ValueError(
                            f"{count} blank value(s) remain in {col}. No previous value "
                            "was available to copy, so the script stopped without inserting rows."
                        )

                # train_no should be an integer.
                if ((df["train_no"] % 1) != 0).any():
                    raise ValueError("train_no contains a non-integer value; no rows were inserted.")
                df["train_no"] = df["train_no"].astype("int64")

                rows = [
                    tuple(
                        int(value) if col == "train_no" else float(value)
                        for col, value in zip(db_columns, row)
                    )
                    for row in df[db_columns].itertuples(index=False, name=None)
                ]

                insert_query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
                    table_ident,
                    sql.SQL(", ").join(sql.Identifier(c) for c in db_columns),
                ).as_string(conn)

                execute_values(cur, insert_query, rows, page_size=5000)
                inserted = len(rows)

        print(f"Success: appended {inserted:,} rows to {SCHEMA}.{TABLE}.")
        print("Only x_g, y_g, z_g and train_no were inserted; existing rows were not updated or deleted.")
        print("Run this script only once for this file, or it will append the rows again.")

    finally:
        conn.close()


if __name__ == "__main__":
    try:
        append_data()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
