import time
import os
from pymongo import MongoClient

def dbLogger(func):
    def f(self, *args, **kwargs):
        if self.DB is not None:
            return func(self, *args, **kwargs)
        return None
    return f

def canLog(log_file = 'default.txt'):
    """Decorator for enabling something for logging. Should only be used in classes that extend logger"""
    def what_am_I_doing_with_my_life(func):
        def logEnabled(self, *args, **kwargs):
            if self.log_dir is not None:
                fp = os.path.join(self.log_dir, log_file)

                old = self.fout
                self.fout = self.open_file(fp)
            
                self.log('{} Logging from {}'.format(time.time(), func.__name__))
                val = func(self, *args, **kwargs)

                self.close_file(fp)
                self.fout = old
                return val
            else:
                return func(self, *args, **kwargs)
            return
        return logEnabled
    return what_am_I_doing_with_my_life

class logger(object):
    """gives some basic logging functionality"""

    def __init__(self, params, creationInstant = None):
        self.log_dir = params.get('log_dir')
        if self.log_dir is not None and not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        self.handlers = {}
        self.fout = None
        self.CreationInstant = time.time() if creationInstant is None else creationInstant
        
        dbURI = params.get('dbURI')
        dbName = params.get('dbName')
        
        self.DB = None
        if dbURI is not None and dbName is not None:
            client = MongoClient(dbURI)
            self.DB = client[dbName]

        return
    
    @dbLogger
    def logConfig(self, config):
        configObj = {
            'CreationTime' : self.CreationInstant
            }
        if not self.DB.Configs.find_one(configObj):
            configObj['Config'] = config
            self.DB.Configs.insert_one(configObj)

        return

    @dbLogger
    def logDecision(self, move_num, game_id, state, decision, probabilities, isTrianing):
        decisionObj = {
                'CreationInstant' : self.CreationInstant,
                'GameId' : game_id,
                'MoveNum' : move_num,
                'State' : state,
                'Decision' : decision,
                'Probabilities' : probabilities,
                'IsTraining' : isTrianing
            }
        self.DB.Decisions.insert_one(decisionObj)
        return


    #Local Logging

    def log(self, s, end = '\n'):
        if self.fout is not None:
            self.fout.write(s + end)
        return

    def open_file(self, fp, mode = 'a'):
        if fp in self.handlers:
            h = self.handlers[fp]
        else:
            h = (0, open(fp, mode)) # sucker for code reuse.

        self.handlers[fp] = (h[0] + 1, h[1]) 
        return h[1]

    def close_file(self, fp):
        assert fp in self.handlers and self.handlers[fp][0] > 0, 'Cannot close file that is not open {}'.format(fp)

        h = self.handlers[fp]
        if h[0] == 1:
            h[1].close()
            del self.handlers[fp]
        else:
            self.handlers[fp] = (h[0]-1, h[1])
        return




