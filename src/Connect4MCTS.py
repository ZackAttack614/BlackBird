from FixedMCTS import FixedMCTS as MCTS_base
import numpy as np

class BoardState():
    def __init__(self, dims, player, **kwargs):
        self.Board = np.zeros(dims)
        self.Player = player
        return super().__init__(**kwargs)

    def Copy(self):
        copy = BoardState(self.Board.shape, self.Player)
        copy.Board = np.copy(self.Board)
        return copy

    def __str__(self):
        array = np.flipud(self.Board.transpose())
        s = ''
        for i in range(array.shape[0]):
            s += '[ '
            for j in range(array.shape[1]):
                if array[i,j] == 1:
                    s += ' X '
                elif array[i,j] == 2:
                    s += ' O '
                else:
                    s += '   '
                if j < array.shape[1] - 1:
                    s += '|'
            s += ']\n\n'

        return s

    def __eq__(self, other):
        if other.Player != self.Player:
            return False
        return (other.Board == self.Board).all()

    def __hash__(self):
        return "{0}{1}".format(self.Player,str(self)).__hash__()

class Connect4MCTS(MCTS_base):

    def __init__(self, maxDepth, explorationRate, width = 7, height = 6, inARow = 4, threads = 1, **kwargs):
        self.Width = width
        self.Height = height
        self.InARow = inARow
        self.Dirs = [(0,1),(1,1),(1,0),(1,-1)]
        return super().__init__(maxDepth, explorationRate, threads, **kwargs)


    # Overriding from MCTS
    def LegalActions(self, state):
        return np.array([1 if state.Board[col, self.Height-1] == 0 else 0 for col in range(self.Width)])
    
    def ApplyAction(self, state, action):
        next = state.Copy()
        next.Player = 1 if state.Player == 2 else 2
        for row in range(self.Height):
            if next.Board[action, row] == 0:
                next.Board[action, row] = state.Player
                break
        return next
    
    def NewGame(self):
        return BoardState((self.Width, self.Height), 1)

    def Winner(self, state, prevAction = None):
        board = state.Board

        if prevAction is not None:
            for j in reversed(range(self.Height)):
                if board[prevAction, j] != 0:
                    win = self.__checkVictory(board, prevAction, j)
                    if win is not None:
                        return win

        if self.__isDraw(board):
            return 0

        if prevAction is None:
            for i in range(self.Width):
                for j in range(self.Height):
                    if board[i,j] == 0:
                        continue
                    win = self.__checkVictory(board, i, j)
                    if win is not None: 
                        return win

        return None


    # Private Functions
    def __isDraw(self,board):
        for i in range(self.Width):
            if board[i, self.Height-1] == 0:
                return False
        return True

    def __checkVictory(self, board, i, j):
        p = board[i,j]
        for dir in self.Dirs:
            inARow = 0
            r = 0
            while r*dir[0] + i < self.Width and r*dir[1] + j < self.Height and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r += 1
            r = -1
            while r*dir[0] + i >= 0 and r*dir[1] + j < self.Height and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r -= 1
            if inARow >= self.InARow:
                return p
        return None







