import sqlite3
import os

class Connection(object):
    def __init__(self, isLocal=True):
        directory = os.path.join(__file__, '../data')

        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._conn = sqlite3.connect(os.path.join(directory, 'blackbird.db'))
        self.cursor = self._conn.cursor()
        self._makeSchema()

    def GetGame(self):
        pass

    def PutGame(self, game):
        pass

    def PutTrainingStatistic(self, statistic):
        pass

    def PutModel(self, model):
        pass

    def _makeSchema(self):
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
                FOREIGN KEY(ArchitectureKey) REFERENCES ArchitectureDim(ArchitectureKey));""")

        tableStatements.append("""
            CREATE TABLE ConfigurationDim(
                ConfigurationKey INTEGER PRIMARY KEY NOT NULL,
                ConfigJSON TEXT);""")

        tableStatements.append("""
            CREATE TABLE TrainingStatisticsFact(
                TrainingStatisticsKey INTEGER PRIMARY KEY NOT NULL,
                ModelKey INTEGER NOT NULL,
                ConfigurationKey INTEGER NOT NULL,
                Time REAL,
                Loss REAL,
                WinsVsRandom INTEGER,
                DrawsVsRandom INTEGER,
                LossesVsRandom INTEGER,
                WinsVsSelf INTEGER,
                DrawsVsSelf INTEGER,
                LossesVsSelf INTEGER,
                FOREIGN KEY (ModelKey) REFERENCES ModelDim(ModelKey),
                FOREIGN KEY (ConfigurationKey) REFERENCES ConfigurationDim(ConfigurationKey));""")

        tableStatements.append("""
            CREATE TABLE JobQueue(
                JobKey INTEGER PRIMARY KEY NOT NULL,
                QueueTime INTEGER,
                StartTime INTEGER,
                EndTime INTEGER,
                RPC INTEGER,
                PID INTEGER);""")

        for command in tableStatements:
            self.cursor.execute(command)

    def _close(self):
        pass

    def __del__(self):
        self._close()

    def __enter__(self):
        pass       
    def __exit__(self):
        pass

if __name__ == '__main__':
    conn = Connection()
    del conn