from app.vector_store import get_connection

with get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("SELECT current_database();")
        result = cursor.fetchone()

        print("Connected to: ", result[0])
