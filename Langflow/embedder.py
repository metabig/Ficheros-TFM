import psycopg2
import argparse
from openai import OpenAI

def create_embeddings(db_params, table_name, column_name, api_key, batch_size=100, vector_dim=1536):
    """
    Connects to a PostgreSQL database and adds a vector column with embeddings for the specified column.

    Args:
        db_params (dict): Database connection parameters (host, port, dbname, user, password).
        table_name (str): Name of the table to process.
        column_name (str): Name of the column containing text to embed.
        api_key (str): OpenAI API key.
        batch_size (int): Number of rows to process per batch (default: 100).
        vector_dim (int): Dimension of the vector embeddings (default: 1536 for text-embedding-3-small).
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    try:
        # Establish database connection
        with psycopg2.connect(**db_params) as conn:
            # Verify table and column existence
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
                if not cur.fetchone()[0]:
                    print(f"Table {table_name} does not exist.")
                    return

                cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)", 
                            (table_name, column_name))
                if not cur.fetchone()[0]:
                    print(f"Column {column_name} does not exist in table {table_name}.")
                    return

                # Identify the primary key
                cur.execute("""
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.constraint_type = 'PRIMARY KEY' AND tc.table_name = %s
                """, (table_name,))
                pk_columns = cur.fetchall()
                if not pk_columns:
                    print(f"No primary key found for table {table_name}.")
                    return
                pk_column = pk_columns[0][0]

                # Add vector column for embeddings if it doesn’t exist
                embedding_column = f"{column_name}_embeddings"
                cur.execute(f"ALTER TABLE {table_name} ADD COLUMN IF NOT EXISTS {embedding_column} vector({vector_dim})")
                conn.commit()

            # Process rows in batches using a cursor
            with conn.cursor() as fetch_cur:
                fetch_cur.execute(f"""
                    DECLARE embedding_cursor CURSOR WITH HOLD FOR 
                    SELECT {pk_column}, {column_name} FROM {table_name} WHERE {embedding_column} IS NULL
                """)
                conn.commit()  # Commit to make the cursor available

                with conn.cursor() as update_cur:
                    total_processed = 0
                    while True:
                        fetch_cur.execute(f"FETCH {batch_size} FROM embedding_cursor")
                        rows = fetch_cur.fetchall()
                        if not rows:
                            break
                        for row in rows:
                            id, text = row
                            if text is None:
                                continue
                            try:
                                # Generate embedding using OpenAI API
                                response = client.embeddings.create(input=[text], model="text-embedding-3-small")
                                embedding = response.data[0].embedding
                                # Update the row with the vector embedding
                                update_cur.execute(
                                    f"UPDATE {table_name} SET {embedding_column} = %s WHERE {pk_column} = %s",
                                    (embedding, id)
                                )
                            except Exception as e:
                                print(f"Error processing row with {pk_column} = {id}: {e}")
                        conn.commit()  # Commit batch updates
                        total_processed += len(rows)
                        print(f"Processed {len(rows)} rows in this batch, total: {total_processed}")

                fetch_cur.execute("CLOSE embedding_cursor")
                conn.commit()

        print("Embedding generation completed.")

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Parse command-line arguments and run the embedding creation process."""
    parser = argparse.ArgumentParser(description="Generate embeddings for a database column using OpenAI.")
    parser.add_argument("--host", required=True, help="Database host")
    parser.add_argument("--port", default="5432", help="Database port")
    parser.add_argument("--dbname", required=True, help="Database name")
    parser.add_argument("--user", required=True, help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--table", required=True, help="Table name")
    parser.add_argument("--column", required=True, help="Column name")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--batch-size", type=int, default=2000, help="Batch size for processing")

    args = parser.parse_args()

    db_params = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password
    }

    create_embeddings(db_params, args.table, args.column, args.api_key, args.batch_size)

if __name__ == "__main__":
    main()