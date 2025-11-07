import psycopg2
import os
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    conn = psycopg2.connect(**get_engine_from_env())
    logger.info("Database connection established.")
    return conn


def create_features_schema(connection, table_name: str): 

    schema = f"""CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            files TEXT[],
            category VARCHAR(100)
            UNIQUE(name)
            );"""

    try:
        cursor = connection.cursor()

        # Execute the schema creation SQL
        cursor.execute(schema)
        
        # Commit the changes
        connection.commit() 

        print("Database schema created successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # Close the cursor and connection
        if cursor:
            cursor.close()
        if connection:
            connection.close()


def Update_features_schema(connection, table_name: str): 
    alter_statements = [
        f"ALTER TABLE {table_name} ADD CONSTRAINT unique_name UNIQUE (name);",
    ]

    try:
        cursor = connection.cursor()

        for statement in alter_statements:
            cursor.execute(statement)
        
        connection.commit() 

        print("Database schema updated successfully.")
        
    except Exception as e:
        print(f"An error occurred during schema update: {e}")
        
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# use arguments parser to parse command line arguments for connection parameters
if __name__ == "__main__":
    connection_params = create_connection()
    # create_features_schema(connection_params, "features")
    Update_features_schema(connection_params, "features")