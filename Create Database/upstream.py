import psycopg2
import psycopg2.extras
import os
import logging
import csv
import ast
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================
# Database connection setup
# =============================================================

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
# Insert features from CSV
# =============================================================

def _parse_files_column(value: str):
    if not value:
        return None
    value = value.strip()
    # many rows store arrays like "['A','B']" (with quotes). Try literal_eval first.
    try:
        parsed = ast.literal_eval(value)
        # ensure list or tuple -> convert to list
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        # sometimes it's a string containing the list representation; return as single-element list
        return [str(parsed)]
    except Exception:
        # try to remove outer quotes if present then literal_eval
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            try:
                parsed = ast.literal_eval(value[1:-1])
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [str(parsed)]
            except Exception:
                pass
        # fallback: try to split on non-word chars to recover items
        import re
        items = re.findall(r"[\w\.\-]+", value)
        return items if items else None

def insert_features_from_csv(csv_path: str, table_name: str = "features"):
    conn = create_connection()
    if conn is None:
        logger.error("No DB connection. Aborting insert.")
        return

    rows = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                # CSV columns: key_name, description, files, category
                name = (r.get("key_name") or r.get("key name") or "").strip()
                if not name:
                    continue
                description = (r.get("description") or "").strip() or None
                files_raw = r.get("files") or ""
                files = _parse_files_column(files_raw)
                category = (r.get("category") or "").strip() or None
                rows.append((name, description, files, category))

        if not rows:
            logger.info("No rows parsed from CSV.")
            return

        with conn.cursor() as cur:
            # use execute_values for bulk insert
            insert_sql = f"INSERT INTO {table_name} (name, description, files, category) VALUES %s ON CONFLICT (name) DO NOTHING"
            psycopg2.extras.execute_values(
                cur, insert_sql, rows, template="(%s, %s, %s, %s)"
            )
            conn.commit()
            logger.info("Inserted %d feature rows into '%s'.", len(rows), table_name)

    except Exception as e:
        logger.exception("Failed to insert features: %s", e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

# =============================================================
# Main execution
# =============================================================

if __name__ == "__main__":
    default_csv = os.path.join(os.path.dirname(__file__), "Relevent Features.csv")
    if not os.path.exists(default_csv):
        logger.error("Default CSV not found: %s", default_csv)
    else:
        insert_features_from_csv(default_csv, "features")