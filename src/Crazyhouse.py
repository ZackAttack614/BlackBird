from GameState import GameState
from proto.state_pb2 import State

import json
import numpy as np

class BoardState(GameState):

    PieceTypes = {
        0 : ' ',
        1 : 'K',
        2 : 'Q',
        3 : 'R',
        4 : 'B',
        5 : 'N',
        6 : 'P',
        -1 : 'k',
        -2 : 'q',
        -3 : 'r',
        -4 : 'b',
        -5 : 'n',
        -6 : 'p',
        7 : "O-O",
        -7 : "o-o",
        8 : "O-O-O",
        -8 : "o-o-o",
        # maybe something for promoting pawns
        
        }
    # Players = {0: ' ', 1 : 'X', 2 : 'O'}
    # Dirs = [(0,1),(1,1),(1,0),(1,-1)]
    BoardShape = np.array([8, 8], dtype=np.int8)
    GameType = 'Crazyhouse'

    def __init__(self):
        self.Board = np.zeros((8, 8, 2), dtype=np.int8)
        self.WhiteHand = []
        self.BlackHand = []
        self.Player = 1
        self.PreviousPlayer = None

    def Copy(self):
        copy = BoardState()
        copy.Player = self.Player
        copy.Board = np.copy(self.Board)
        copy.WhiteHand = self.WhiteHand
        copy.BlackHand = self.BlackHand
        return copy
    
    def LegalActions(self):
        return None
    def LegalActionShape(self):
        return None
    def ApplyAction(self, action):
        return None
    def Winner(self, prevAction = None):
        return None
    def NumericRepresentation(self):
        return None
    def __str__(self):


    if __name__ == '__main__':
        state = BoardState()