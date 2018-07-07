import sqlite3
import uuid
import os

class Connection(object):
    def __init__(self, isLocal=True):
        directory = 'data' # Placeholder until I can figure out a robust method

        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._conn = sqlite3.connect(os.path.join(directory, 'blackbird.db'))
        self.Cursor = self._conn.cursor()
        self._makeSchema(self.Cursor)
        self.ModelKey = None

    def PutArchitecture(self, arch):
        command = 'INSERT INTO ArchitectureDim(ArchitectureJSON) VALUES (?);'
        self.Cursor.execute(command, (arch,))
        self._conn.commit()

    def PutModel(self, archKey, gameType, name):
        command = 'INSERT INTO ModelDim(ArchitectureKey, GameType, Name) VALUES(?,?,?);'
        self.Cursor.execute(command, (archKey, gameType, name))
        self.ModelKey = self.Cursor.lastrowid
        self._conn.commit()

    def GetGames(self):
        command = 'SELECT State FROM GameStateFact WHERE ModelKey = ?;'
        self.Cursor.execute(command, (self.ModelKey,))
        return [state[0] for state in self.Cursor.fetchall()]

    def PutGames(self, gameType, games):
        command = 'INSERT INTO GameStateFact(ModelKey, GameType, State) VALUES (?, ?, ?);'
        self.Cursor.executemany(command, [(self.ModelKey, gameType, g) for g in games])
        self._conn.commit()

    def PutTrainingStatistic(self, statistic):
        pass

    def Close(self):
        self._conn.close()

    def SetLastModel(self, gameType, name):
        self.Cursor.execute('SELECT MAX(ModelKey) FROM ModelDim WHERE GameType = ?;', (gameType,))
        key = self.Cursor.fetchone()

        if any(key):
            self.ModelKey = key[0]
        else:
            self.PutModel(1, gameType, name)

    def _makeSchema(self, cursor):
        check = """SELECT COUNT(name) FROM sqlite_master
                   WHERE type='table' AND name='TrainingStatisticsFact';"""
        cursor.execute(check)
        if cursor.fetchone()[0] == 1:
            return

        tableStatements = []
        tableStatements.append("""
            CREATE TABLE ArchitectureDim(
                ArchitectureKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ArchitectureJSON TEXT);""")

        tableStatements.append("""
            CREATE TABLE ModelDim(
                ModelKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ArchitectureKey INTEGER NOT NULL,
                GameType TEXT NOT NULL,
                Name TEXT,
                FOREIGN KEY(ArchitectureKey)
                    REFERENCES ArchitectureDim(ArchitectureKey));""")

        tableStatements.append("""
            CREATE TABLE ConfigurationDim(
                ConfigurationKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ConfigJSON TEXT);""")

        tableStatements.append("""
            CREATE TABLE TrainingStatisticsFact(
                TrainingStatisticsKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelKey INTEGER NOT NULL,
                ConfigurationKey INTEGER NOT NULL,
                Rating REAL,
                Time REAL,
                Loss REAL,
                WinsVsRandom INTEGER,
                DrawsVsRandom INTEGER,
                LossesVsRandom INTEGER,
                WinsVsSelf INTEGER,
                DrawsVsSelf INTEGER,
                LossesVsSelf INTEGER,
                FOREIGN KEY (ModelKey)
                    REFERENCES ModelDim(ModelKey),
                FOREIGN KEY (ConfigurationKey)
                    REFERENCES ConfigurationDim(ConfigurationKey));""")

        tableStatements.append("""
            CREATE TABLE JobQueue(
                JobKey INTEGER PRIMARY KEY AUTOINCREMENT,
                QueueTime INTEGER,
                StartTime INTEGER,
                EndTime INTEGER,
                RPC INTEGER,
                PID INTEGER);""")

        tableStatements.append("""
            CREATE TABLE GameStateFact( 
                GameStateKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelKey INTEGER NOT NULL,
                GameType TEXT NOT NULL,
                State BYTES NOT NULL,
                FOREIGN KEY (ModelKey)
                    REFERENCES ModelDim(ModelKey));""")

        for command in tableStatements:
            cursor.execute(command)

    def __del__(self):
        self.Close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Close()
