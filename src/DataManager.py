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
        self.ModelKey = None

    def PutModel(self, gameType, name, version):
        command = 'INSERT INTO ModelDim(GameType, Name, Version) VALUES(?,?,?);'
        self.Cursor.execute(command, (gameType, name, version))
        self.ModelKey = self.Cursor.lastrowid
        self.Cursor.execute("""INSERT INTO TrainingStatisticsFact(
            ModelKey) VALUES (?);""", (self.ModelKey,))

        self._conn.commit()

    def GetGames(self):
        command = 'SELECT State FROM GameStateFact WHERE ModelKey = ?;'
        self.Cursor.execute(command, (self.ModelKey,))
        return [state[0] for state in self.Cursor.fetchall()]

    def PutGames(self, gameType, games):
        command = 'INSERT INTO GameStateFact(ModelKey, GameType, State) VALUES (?, ?, ?);'
        self.Cursor.executemany(command, [(self.ModelKey, gameType, g) for g in games])
        self._conn.commit()

    def DumpToZip(self, modelKey):
        self.Cursor.execute('SELECT State FROM GameStateFact WHERE ModelKey = ?;', (modelKey,))

        with gzip.open('data/states_{0}.gz'.format(modelKey), 'wb') as states_out:
            for state in self.Cursor.fetchall():
                states_out.write(state[0]+b'\x07\x07\x07')

    def PutTrainingStatistic(self, statType, stats, modelKey):
        command = """UPDATE TrainingStatisticsFact
            SET {0} = ?, {1} = ?, {2} = ?
            WHERE ModelKey = ?;"""
            
        if statType == 'random':
            command = command.format('WinsVsRandom', 'DrawsVsRandom', 'LossesVsRandom')
        elif statType == 'self':
            command = command.format('WinsVsSelf', 'DrawsVsSelf', 'LossesVsSelf')
        elif statType == 'MCTS':
            command = command.format('WinsVsMCTS', 'DrawsVsMCTS', 'LossesVsMCTS')
        else:
            raise ValueError('statType must be "random", "self", or "MCTS".')

        self.Cursor.execute(command, (*stats.values(), modelKey))
        self._conn.commit()

    def Close(self):
        self._conn.close()

    def SetLastModel(self, gameType, name):
        self.Cursor.execute('SELECT MAX(ModelKey) FROM ModelDim WHERE GameType = ?;', (gameType,))
        key = self.Cursor.fetchone()

        if any(key):
            self.ModelKey = key[0]
        else:
            self.PutModel(gameType, name, 1)

        self.Cursor.execute('SELECT Version FROM ModelDim WHERE ModelKey = ?;', (self.ModelKey,))
        return self.Cursor.fetchone()[0]

    def _makeSchema(self, cursor):
        check = """SELECT COUNT(name) FROM sqlite_master
                   WHERE type='table' AND name='TrainingStatisticsFact';"""
        cursor.execute(check)
        if cursor.fetchone()[0] == 1:
            return

        tableStatements = []

        tableStatements.append("""
            CREATE TABLE ModelDim(
                ModelKey INTEGER PRIMARY KEY AUTOINCREMENT,
                GameType TEXT NOT NULL,
                Name TEXT,
                Version INTEGER DEFAULT 1);""")

        tableStatements.append("""
            CREATE TABLE ConfigurationDim(
                ConfigurationKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ConfigJSON TEXT);""")

        tableStatements.append("""
            CREATE TABLE TrainingStatisticsFact(
                TrainingStatisticsKey INTEGER PRIMARY KEY AUTOINCREMENT,
                ModelKey INTEGER NOT NULL,
                ConfigurationKey INTEGER,
                Rating REAL,
                Time REAL,
                Loss REAL,
                WinsVsRandom INTEGER,
                DrawsVsRandom INTEGER,
                LossesVsRandom INTEGER,
                WinsVsSelf INTEGER,
                DrawsVsSelf INTEGER,
                LossesVsSelf INTEGER,
                WinsVsMCTS INTEGER,
                DrawsVsMCTS INTEGER,
                LossesVsMCTS INTEGER,
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
