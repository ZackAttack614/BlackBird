import numpy as np

class node:
    def __init__(self, state, prior, c_PUCT, move=None, parent=None):
        self.state = state
        self.move = move
        self.parent = parent
        self.children = []
        
        self.c_PUCT = c_PUCT
        
        self.W = 0   # Total action value
        self.N = 0   # Visit count
        self.U = 0   # PUCT value
        self.Q = 0   # Mean action value
        
        self.P = prior   # Prior probability
        
    def backup(self, reward):
        """ Backup values obtained from mcts search through to the root.
        """
        self.N += 1
        self.W += reward
        self.Q = self.W / self.N
        
        if self.parent is not None:
            self.parent.backup(reward)

    def getU(self):
        if self.parent is None:
            return 0
        return (self.c_PUCT * self.P * np.sqrt(self.parent.N)) / (1 + self.N) # Technically that is not sum(child.N) because we visit the parent once before the children. but this makes more sense?
