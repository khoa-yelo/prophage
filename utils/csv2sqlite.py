#!/usr/bin/env python3
import argparse
import logging
import sqlite3
import sys

import pandas as pd

def setup_logging(level: str) -> None:
    """
    Configure root logger with the given level.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

def df_to_sqlite_with_indexes(df: pd.DataFrame, db_path: str, table_name: str) -> None:
    """
    Write a DataFrame to SQLite and create an index on each column.

    :param df: pandas DataFrame to store
    :param db_path: path to the SQLite database file (will be created if it doesn't exist)
    :param table_name: name of the table to create/replace
    """
    logging.info("Connecting to SQLite database at '%s'", db_path)
    conn = sqlite3.connect(db_path)
    try:
        logging.info("Writing DataFrame to table '%s' (if exists, it will be replaced)", table_name)
        df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
        
        cursor = conn.cursor()
        for col in df.columns:
            idx_name = f"idx_{table_name}_{col}".replace(" ", "_")
            sql = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({col});"
            logging.debug("Creating index '%s' on column '%s'", idx_name, col)
            cursor.execute(sql)
        
        conn.commit()
        logging.info("All indexes created successfully")
    except Exception as e:
        logging.exception("Error while writing table or creating indexes")
        sys.exit(1)
    finally:
        conn.close()
        logging.info("Closed SQLite connection")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a CSV file into a SQLite database table and index all columns."
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file to load into SQLite."
    )
    parser.add_argument(
        "db_path",
        help="Path to the SQLite database file (will be created if it doesn't exist)."
    )
    parser.add_argument(
        "table_name",
        help="Name of the table to create/replace in the database."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    setup_logging(args.log_level)

    logging.info("Reading CSV file '%s'", args.input_csv)
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        logging.exception("Failed to read CSV file")
        sys.exit(1)

    df_to_sqlite_with_indexes(df, args.db_path, args.table_name)
    logging.info(
        "Finished: DataFrame from '%s' is now in '%s' (table '%s') with indexes on all columns",
        args.input_csv, args.db_path, args.table_name
    )

if __name__ == "__main__":
    main()
