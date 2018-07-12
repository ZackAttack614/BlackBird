import sqlite3
import uuid
import gzip
import os

class Connection(object):
    def __init__(self, isLocal=True):
        directory = 'data' # Placeholder until I can figure out a robust method

        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._conn = sqlite3.connect(os.path.join(directory, 'blackbird.db'))
        self.Cursor = self._conn.cursor()
        self._makeSchema(self.Cursor)

    def GetLastVersion(self, gameType, name):
        self.Cursor.execute("""
            SELECT Version FROM ModelDim
            WHERE Name = ? AND GameType = ?
            ORDER BY Version DESC LIMIT 1;""", (name, gameType))
        version = self.Cursor.fetchone()
        if version is None:
            self.PutModel(gameType, name, 1)
            version = [1]
        return version[0]

    def PutModel(self, gameType, name, version):
        self.Cursor.execute("""INSERT INTO ModelDim(GameType, Name, Version)
            VALUES(?, ?, ?);""", (gameType, name, version))
        self._conn.commit()

    def GetGames(self, name, version):
        self.Cursor.execute("""
            SELECT State FROM GameStateFact 
            WHERE ModelKey = (
                SELECT ModelKey FROM ModelDim
                WHERE Name = ? AND Version = ?
                ORDER BY ModelKey DESC LIMIT 1);""", (name, version))
        return [state[0] for state in self.Cursor.fetchall()]

    def PutGames(self, name, version, gameType, games):
        self.Cursor.executemany("""
            INSERT INTO GameStateFact(ModelKey, GameType, State)
            VALUES(
                (SELECT ModelKey FROM ModelDim WHERE Name = ? AND Version = ?
                 ORDER BY ModelKey DESC LIMIT 1),
                ?,
                ?);""",
            [(name, version, gameType, g) for g in games])
        self._conn.commit()

    def DumpToZip(self, name, version):
        with gzip.open('data/states_{0}_v{1}.gz'.format(name, version), 'wb') as states_out:
            for state in self.GetGames(name, version):
                states_out.write(state[0] + b'\x07\x07\x07')

    def PutTrainingStatistic(self, result, name, version, opName, opVersion=0):
        self.Cursor.execute("""
            INSERT INTO TrainingStatisticsFact(
                ModelKey, OpponentKey, Result)
            VALUES(
                (SELECT ModelKey FROM ModelDim WHERE Name = ? AND Version = ?),
                (SELECT ModelKey FROM ModelDim WHERE Name = ? AND Version = ?),
                ?
            );""", (name, version, opName, opVersion, result))
        self._conn.commit()

    def Close(self):
        self._conn.close()

    def _makeSchema(self, cursor):
        for _ in cursor.execute("""SELECT name FROM sqlite_master
                   WHERE type='table' AND name='TrainingStatisticsFact';"""):
            return

        self.Cursor.executescript("""
            CREATE TABLE ModelDim(
                ModelKey INTEGER PRIMARY KEY AUTOINCREMENT,
                GameType TEXT,
                Name TEXT,
                Version INTEGER DEFAULT 1);
                
            CREATE TABLE ConfigurationDim(
                ConfigurationKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ConfigJSON TEXT);
                
            CREATE TABLE TrainingStatisticsFact(
                TrainingStatisticsKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelKey INTEGER NOT NULL,
                OpponentKey INTEGER NOT NULL,
                Result INTEGER,
                Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ModelKey)
                    REFERENCES ModelDim(ModelKey),
                FOREIGN KEY (OpponentKey)
                    REFERENCES ModelDim(ModelKey));
                    
            CREATE TABLE JobQueue(
                JobKey INTEGER PRIMARY KEY AUTOINCREMENT,
                QueueTime INTEGER,
                StartTime INTEGER,
                EndTime INTEGER,
                RPC INTEGER,
                PID INTEGER);
                
            CREATE TABLE GameStateFact( 
                GameStateKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelKey INTEGER NOT NULL,
                GameType TEXT NOT NULL,
                State BYTES NOT NULL,
                FOREIGN KEY (ModelKey)
                    REFERENCES ModelDim(ModelKey));""")

        self.Cursor.executescript("""
            INSERT INTO ModelDim(Name, Version) VALUES('RANDOM', 0);
            INSERT INTO ModelDim(Name, Version) VALUES('MCTS', 0);""")

    def __del__(self):
        self.Close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Close()
