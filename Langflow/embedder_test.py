import psycopg2
import argparse
from openai import OpenAI

def semantic_search(db_params, table_name, column_name, search_phrase, api_key, limit=5):
    """
    Performs a semantic search on a PostgreSQL table with vector embeddings.

    Args:
        db_params (dict): Database connection parameters (host, port, dbname, user, password).
        table_name (str): Name of the table to search.
        column_name (str): Name of the text column (original text).
        search_phrase (str): Phrase to search for.
        api_key (str): OpenAI API key.
        limit (int): Number of top results to return (default: 5).
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Connect to the database
    try:
        conn = psycopg2.connect(**db_params)
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return

    # Generate embedding for the search phrase
    try:
        response = client.embeddings.create(input=[search_phrase], model="text-embedding-3-small")
        search_embedding = response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding for search phrase: {e}")
        conn.close()
        return

    # Create a cursor for the query
    cur = conn.cursor()

    # Check if the table and columns exist
    embedding_column = f"{column_name}_embeddings"
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)", (table_name,))
    table_exists = cur.fetchone()[0]
    if not table_exists:
        print(f"Table {table_name} does not exist.")
        conn.close()
        return

    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)", 
                (table_name, column_name))
    text_column_exists = cur.fetchone()[0]
    cur.execute("SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s)", 
                (table_name, embedding_column))
    embedding_column_exists = cur.fetchone()[0]
    if not text_column_exists or not embedding_column_exists:
        print(f"One or both columns ({column_name}, {embedding_column}) do not exist in table {table_name}.")
        conn.close()
        return

    # Perform the semantic search using cosine similarity (<=> operator in pgvector)
    query = f"""
        SELECT {column_name}, 1 - ({embedding_column} <=> %s::vector) AS similarity
        FROM {table_name}
        WHERE {embedding_column} IS NOT NULL
        ORDER BY {embedding_column} <=> %s::vector
        LIMIT %s
    """
    try:
        cur.execute(query, (search_embedding, search_embedding, limit))
        results = cur.fetchall()
    except psycopg2.Error as e:
        print(f"Error executing search query: {e}")
        cur.close()
        conn.close()
        return

    # Display results
    print(f"\nTop {limit} matches for '{search_phrase}':")
    print("--------------------------------------------------")
    for i, (text, similarity) in enumerate(results, 1):
        print(f"{i}. Text: {text}")
        print(f"   Similarity: {similarity:.4f}")
        print("--------------------------------------------------")

    # Clean up
    cur.close()
    conn.close()

def main():
    """Parse command-line arguments and run the semantic search."""
    parser = argparse.ArgumentParser(description="Perform semantic search on a PostgreSQL table with vector embeddings.")
    parser.add_argument("--host", required=True, help="Database host")
    parser.add_argument("--port", default="5432", help="Database port")
    parser.add_argument("--dbname", required=True, help="Database name")
    parser.add_argument("--user", required=True, help="Database user")
    parser.add_argument("--password", required=True, help="Database password")
    parser.add_argument("--table", required=True, help="Table name")
    parser.add_argument("--column", required=True, help="Text column name")
    parser.add_argument("--phrase", required=True, help="Search phrase")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    db_params = {
        "host": args.host,
        "port": args.port,
        "dbname": args.dbname,
        "user": args.user,
        "password": args.password
    }

    semantic_search(db_params, args.table, args.column, args.phrase, args.api_key, args.limit)

if __name__ == "__main__":
    main()