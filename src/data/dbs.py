import csv
import sqlite3
from plyvel import plyvel #type: ignore
from typing import List, Tuple


from JargonTTS.src.data.base import BaseDB

class SQLiteDBStore(BaseDB):
    def __init__(self, path: str):
        
        self.path = path
        self.conn = sqlite3.connect(path)
        self.cursor = self.conn.cursor()
        self.command_pairs()
        self.cursor.execute(self.commands['create'])

    def command_pairs(self) -> None:
        """
        Defines SQL commands for creating and inserting data into the 'jargon' table.
        """
        self.commands = {
            'create': 'CREATE TABLE IF NOT EXISTS jargon (term TEXT, definition TEXT)',
            'insert': 'INSERT INTO jargon (term, definition) VALUES (?, ?)',
            'select': 'SELECT term, definition FROM jargon'
        }

    def add_to_db(self, rows: List[Tuple[str, str]], commit_once: bool = True) -> None:
        """
        Inserts multiple rows of jargon terms and definitions into the database.
        Commits once after all rows are inserted if commit_once is True.
        """
        try:
            self.cursor.executemany(self.commands['insert'], rows)
            if commit_once:
                self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error inserting rows: {e}")

    
    def delete_from_db(self, rows):

        return NotImplemented

    def close(self) -> None:
        
        if self.conn:
            self.conn.close()
            print("Database connection closed.")

    def dump_to_csv(self, csv_path: str) -> None:

        try:
            with open(csv_path, 'w', newline='') as csv_file:

                writer = csv.writer(csv_file)
                writer.writerow(['term', 'definition'])
                
                self.cursor.execute("SELECT term, definition FROM jargon")

                rows = self.cursor.fetchall()
                writer.writerows(rows)

            print(f"Data successfully exported to {csv_path}")
            
        except Exception as e:
            print(f"Error exporting to CSV: {e}")

    

class LevelDBStore(BaseDB):
    def __init__(self):
        super().__init__()