import time
import os

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

    def __init__(self, d = None):
        self.log_dir = d
        if d is not None and not os.path.isdir(d):
            os.mkdir(d)
        self.handlers = {}
        self.fout = None
        return

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




