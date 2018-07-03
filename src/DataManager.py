import sqlite3
import os

class Connection(object):
    def __init__(self, isLocal=True):
        directory = os.path.join(__file__, '../data')

        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._conn = sqlite3.connect(os.path.join(directory, 'blackbird.db'))
        self.Cursor = self._conn.cursor()
        self._makeSchema(self.Cursor)

    def GetGame(self):
        pass

    def PutGame(self, game):
        pass

    def PutTrainingStatistic(self, statistic):
        pass

    def PutModel(self, model):
        pass

    def _makeSchema(self, cursor):
        check = """SELECT COUNT(name) FROM sqlite_master
                   WHERE type='table' AND name='TrainingStatisticsFact';"""
        cursor.execute(check)
        if cursor.fetchone()[0] == 1:
            return

        tableStatements = []
        tableStatements.append("""
            CREATE TABLE ArchitectureDim(
                ArchitectureKey INTEGER PRIMARY KEY NOT NULL,
                ArchitectureJSON BLOB);""")

        tableStatements.append("""
            CREATE TABLE ModelDim(
                ModelKey INTEGER PRIMARY KEY NOT NULL,
                ArchitectureKey INTEGER NOT NULL,
                Name TEXT,
                FOREIGN KEY(ArchitectureKey)
                    REFERENCES ArchitectureDim(ArchitectureKey));""")

        tableStatements.append("""
            CREATE TABLE ConfigurationDim(
                ConfigurationKey INTEGER PRIMARY KEY NOT NULL,
                ConfigJSON TEXT);""")

        tableStatements.append("""
            CREATE TABLE TrainingStatisticsFact(
                TrainingStatisticsKey INTEGER PRIMARY KEY NOT NULL,
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
                JobKey INTEGER PRIMARY KEY NOT NULL,
                QueueTime INTEGER,
                StartTime INTEGER,
                EndTime INTEGER,
                RPC INTEGER,
                PID INTEGER);""")

        for command in tableStatements:
            cursor.execute(command)

    def Close(self):
        self._conn.close()

    def __del__(self):
        self.Close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.Close()

if __name__ == '__main__':
    conn = Connection()
    del conn