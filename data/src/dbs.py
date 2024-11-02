import sqlite3
from typing import List

# Local import
from JargonTTS.data.src.base import BaseDB

class SQLiteDBStore(BaseDB):
    def __init__(self, path: str):
        self.path = path
        self.conn = sqlite3.connect(path)
        # Enable returning bytestrings for BLOB fields
        self.conn.text_factory = bytes
        self.cursor = self.conn.cursor()
        self.command_pairs()
        self.cursor.execute(self.commands['create'])

    def command_pairs(self) -> None:
        self.commands = {
            'create': 'CREATE TABLE IF NOT EXISTS jargon (term BLOB)',
            'insert': 'INSERT INTO jargon (term) VALUES (?)',
            'select': 'SELECT term FROM jargon'
        }

    def add_to_db(self, terms: List[str]) -> None:
        # Convert strings to bytes before insertion
        byte_terms = [(term.encode('utf-8'),) for term in terms]
        self.cursor.executemany(self.commands['insert'], byte_terms)
        self.conn.commit()

    def fetch_all(self) -> List[str]:
        self.cursor.execute(self.commands['select'])
        rows = self.cursor.fetchall()
        # Handle the possibility of both string and bytes data
        return [
            term[0].decode('utf-8') if isinstance(term[0], bytes) else term[0]
            for term in rows
        ]

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
