from blackbird import BlackBird
import numpy as np


# Check out Connect4MCTS.py as an example here.
class BoardState():
    players = {0: ' ', 1 : 'X', 2 : 'O'}
    def __init__(self, size = 3, inARow = 3):
        self.Board = np.zeros((size,size,2))
        self.Size = size
        self.InARow = inARow
        self.Player = 1
        self.Dirs = [(0,1),(1,1),(1,0),(1,-1)]
        return 

    def Copy(self):
        copy = BoardState(self.Board.shape[0])
        copy.Player = self.Player
        copy.Board = np.copy(self.Board)
        return copy

    def LegalActions(self):
        actions = np.zeros(self.Size * self.Size)
        for i in range(self.Size):
            for j in range(self.Size):
                if np.sum(self.Board[i, j, :]) == 0:
                    actions[self.CoordsToIndex((i,j))] = 1

        return actions

    def ApplyAction(self, action):
        coords = self.IndexToCoords(action)
        assert np.sum(self.Board[coords[0], coords[1], :]) == 0, 'Ahh. Can\'t go there! {}'.format(action)
        self.Board[coords[0], coords[1], self.Player - 1] = 1
        self.Player = 1 if self.Player == 2 else 2
        return
    
    def Winner(self, prevAction = None):
        board = self.Collapsed()

        if prevAction is not None:
            coords = state.IndexToCoords(prevAction)
            return self.__checkVictory(board, coords[0], coords[1])
        else:
            for i in range(self.Size):
                for j in range(self.Size):
                    if board[i,j] == 0:
                        continue
                    win = self.__checkVictory(board, i, j)
                    if win is not None: 
                        return win

        return None

    def __checkVictory(self, board, i, j):
        p = board[i,j]
        for dir in self.Dirs:
            inARow = 0
            r = 0
            while r*dir[0] + i < self.Size and r*dir[1] + j < self.Size and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r += 1
            r = -1
            while r*dir[0] + i >= 0 and r*dir[1] + j < self.Size and r*dir[1] + j >= 0 and board[r*dir[0] + i, r*dir[1] + j] == p:
                inARow += 1
                r -= 1
            if inARow >= self.InARow:
                return p
        return None
    
    def CoordsToIndex(self, coords):
        return coords[0]*self.Size + coords[1]

    def IndexToCoords(self, index):
        return (index//self.Size, index % self.Size)
    
    def Collapsed(self):
        array = np.zeros(self.Board.shape[:2])
        for p in BoardState.players:
            array[self.Board[:, :, p - 1] == 1] = p
        return array

    def AsInputArray(self):
        player = np.full((self.Size, self.Size), 1 if self.Player == 1 else -1)
        array = np.zeros((1, self.Size, self.Size, 3))
        array[0, :, :, 0:2] = self.Board
        array[0, :, :, 2] = player
        return array

    def __str__(self):
        array = self.Collapsed()
        s = ''
        for i in range(array.shape[0]):
            s += '[ '
            for j in range(array.shape[1]):
                s += ' {} '.format(BoardState.players[array[i, j]])
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

class TicTacToePlayer(BlackBird):
    """Implementation of game"""
    def __init__(self, size = 3, inARow = 3, **kwargs):
        self.Size = size
        self.InARow = inARow
        return super().__init__(**kwargs)

    def LegalActions(self, state):
        return state.LegalActions()

    def ApplyAction(self, state, action):
        next = state.Copy()
        next.ApplyAction(action)
        return next

    def NewGame(self):
        return BoardState(self.Size, self.InARow)

    def Winner(self, state, prevAction = None):
        return state.Winner()

if __name__ == '__main__':
    size = 10
    state = BoardState(size, 5)
    prevAction = None
    while state.Winner(prevAction) is None:
        print(state)
        options = np.where(state.LegalActions() == 1)[0]
        if len(options) == 0:
            break
        action = np.random.choice(options)
        state.ApplyAction(action)
    print(state)
    print(state.Winner())

    print(state.AsInputArray())





