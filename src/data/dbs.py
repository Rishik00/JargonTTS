import sqlite3
from plyvel import plyvel #type: ignore
from JargonTTS.src.data.base import BaseDB

class SQLiteDBStore(BaseDB):
    def __init__(self):
        super().__init__()
    

class LevelDBStore(BaseDB):
    def __init__(self):
        super().__init__()