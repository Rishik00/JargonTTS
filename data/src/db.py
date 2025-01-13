import sqlite3
from typing import List
import os


class SQLiteDB:
    def __init__(self, path: str, db_name: str = 'jargon'):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.text_factory = bytes  # Ensures byte storage
        self.cursor = self.conn.cursor()
        self.db_name = db_name

        # Check if the table exists without recreating it
        self.ensure_table_exists()
        print(f"Connected to database at {path}")

    def ensure_table_exists(self):
        """Ensures the table exists in the database without recreating it."""
        create_query = f'CREATE TABLE IF NOT EXISTS {self.db_name} (term BLOB)'
        self.cursor.execute(create_query)

    def add(self, terms: List[str]) -> None:
        """Add terms to the database."""
        insert_query = f'INSERT INTO {self.db_name} (term) VALUES (?)'
        byte_terms = [(term.encode('utf-8'),) for term in terms]
        self.cursor.executemany(insert_query, byte_terms)
        self.conn.commit()
    
    def getall(self):
        select_query = f"SELECT term FROM {self.db_name}"
        self.cursor.execute(select_query)
        rows = self.cursor.fetchall()

        if not rows:
            return []  # Return empty list if no rows are fetched

        # Decode bytes to strings
        return [
            row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
            for row in rows
        ]

    def fetch(self, limit: int = 1) -> List[str]:
        """Fetch a limited number of rows from the database."""
        select_query = f"SELECT term FROM {self.db_name} LIMIT {limit}"
        self.cursor.execute(select_query)
        rows = self.cursor.fetchall()

        if not rows:
            return []  # Return empty list if no rows are fetched

        # Decode bytes to strings
        return [
            row[0].decode('utf-8') if isinstance(row[0], bytes) else row[0]
            for row in rows
        ]

    def dump_to_txt(self, output_file):
        try:
            with open(output_file, 'w') as f:
                rows = self.getall()
                for term in rows:
                    f.write(term + '\n')  # Ensure each term is written on a new line
        except Exception as e:
            print(f"Error while dumping data to txt: {e}")

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    # Step 1: Print the current working directory
    print("Initial current working directory:", os.getcwd())

    # Step 2: Build the absolute database path
    db_relative_path = r'..\audio_files\jargon_db.db'  # Relative path to database
    db_path = os.path.abspath(db_relative_path)  # Convert to absolute path
    print("Database path:", db_path)

    db = SQLiteDB(db_path)
    try:
        # Fetch and print rows from the existing database
        rows = db.fetch(limit=5)  # Adjust the limit as needed
        db.dump_to_txt(output_file=r'C:\Users\sridh\OneDrive\Desktop\webdev\JargonTTS\data\src\input_file.txt')
        print("Fetched rows:", rows)
    except Exception as e:
        print("Error during database operation:", e)
    finally:
        db.close()
