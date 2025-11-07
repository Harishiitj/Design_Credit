import psycopg2
import psycopg2.extras
import os
import logging
import csv
import ast
from typing import Optional
import pandas as pd
import pyreadstat
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================
# Database connection setup
# =============================================================

"""
SETUP CONNECTION VARIABLES BEFORE RUNNING THE SCRIPT
$env:PGUSER = 'myuser'
$env:PGPASSWORD = 'mypassword'
$env:PGHOST = 'localhost'
$env:PGPORT = '5432'
$env:PGDATABASE = 'mydb'
"""

def create_connection():

    def get_engine_from_env() -> Optional[object]:
        user = os.environ.get("PGUSER")
        password = os.environ.get("PGPASSWORD")
        host = os.environ.get("PGHOST")
        port = os.environ.get("PGPORT") 
        dbname = os.environ.get("PGDATABASE")

        if not (user and password and dbname):
            logger.error(
                "Missing DB configuration. Provide DATABASE_URL or PGUSER/PGPASSWORD/PGDATABASE (and optionally PGHOST/PGPORT)."
            )
            return None
        
        connection_params = {
            "user": user,
            "password": password,
            "host": host or "localhost",
            "port": port or "5432",
            "dbname": dbname,
        }
        return connection_params
    
    params = get_engine_from_env()
    if not params:
        return None

    conn = psycopg2.connect(**params)
    logger.info("Database connection established.")
    return conn

# =============================================================
# Insert features from DTA into separate tables
# =============================================================

DTA_files = ['../Dataset/IABR7EFL.DTA', '../Dataset/IACR7EFL.DTA', '../Dataset/IAHR7EFL.DTA', '../Dataset/IAKR7EFL.DTA', '../Dataset/IAIR7EFL.DTA', '../Dataset/IAMR7EFL.DTA']
def insert_DTA_features(conn, dta_file):

    table_name = os.path.splitext(os.path.basename(dta_file))[0]

    chunk_rows = 5000
    offset = 0
    created = False
    cols = None
    total_inserted = 0

    while True:
        df, meta = pyreadstat.read_dta(dta_file, row_limit=chunk_rows, row_offset=offset)
        if df.shape[0] == 0:
            break

        # Clean column names to be SQL compatible
        df.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df.columns]

        # On first chunk create table and lock down columns (first 10 as before)
        if not created:
            df = df.iloc[:, :10]
            cols = list(df.columns)
            columns_with_types = ', '.join([f'"{col}" TEXT' for col in cols])
            create_table_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_with_types});'
            with conn.cursor() as cursor:
                cursor.execute(create_table_query)
                conn.commit()
            created = True
        else:
            # Ensure subsequent chunks use the same column set (drop extras, pad missing)
            df = df.reindex(columns=cols)

        # Prepare data rows: convert pandas NA to None so psycopg2 inserts NULL
        rows = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in df.itertuples(index=False, name=None)
        ]

        if rows:
            insert_query = f'INSERT INTO "{table_name}" ({", ".join([f"{c}" for c in cols])}) VALUES %s'
            with conn.cursor() as cursor:
                execute_values(cursor, insert_query, rows, page_size=1000)
                conn.commit()
            total_inserted += len(rows)

        offset += len(df)

    logger.info(f"Inserted {total_inserted} rows into table {table_name} from {dta_file}.")

if __name__ == "__main__":
    conn = create_connection()
    # ============================================================
    # Check if connection was successful and check if files exist
    # ============================================================
    if conn:
        for dta_file in DTA_files:
            if os.path.exists(dta_file):
                insert_DTA_features(conn, dta_file)
            else:
                logger.warning(f"Data file {dta_file} does not exist.")
        conn.close()
        