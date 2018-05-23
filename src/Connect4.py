import numpy as np
import random

class Connect4:
    def __init__(self, **kwargs):
        self.Width = kwargs['width'] if ('width' in kwargs) else 5
        self.Height = kwargs['height'] if 'height' in kwargs else 4
        self.InARow = kwargs['inARow'] if 'inARow' in kwargs else 3
        return

    def next_states(self, currentState):
        toMove = currentState[0]
        board = currentState[1]
        nextStates = []
        for col in range(self.Width):
            if board[0,col] != 0:
                continue
            for row in reversed(range(self.Height)):
                if board[row,col] == 0:
                    nextBoard = np.copy(board)
                    nextBoard[row,col] = toMove
                    nextStates.append((-toMove, nextBoard))
                    break

        return nextStates

    def play(self, currentState, col):
        toMove = currentState[0]
        board = currentState[1]
        for row in reversed(range(self.Height)):
            if board[row,col] == 0:
                nextBoard = np.copy(board)
                nextBoard[row, col] = toMove
                nextState = (-toMove, nextBoard)
                break
        return nextState

    def isEnd(self, state):
        dirs = [(0,1),(1,1),(1,0),(1,-1)] # lol. dirs are cols x rows, while we loop the other way.
        board = state[1]
        winner = 0
        countZeros = 0
        for i in range(self.Height):
            for j in range(self.Width):
                if board[i,j] == 0:
                    countZeros += 1
                    continue
                p = board[i,j]
                for dir in dirs:
                    inARow = 0
                    r = 0
                    while r*dir[0] + j < self.Width and r*dir[1] + i < self.Height and r*dir[1] + i >= 0 and board[r*dir[1] + i, r*dir[0] + j] == p:
                        inARow += 1
                        r += 1
                    r = -1
                    while r*dir[0] + j >= 0 and r*dir[1] + i < self.Height and r*dir[1] + i >= 0 and board[r*dir[1] + i, r*dir[0] + j] == p:
                        inARow += 1
                        r -= 1
                    if inARow >= self.InARow:
                        return (1, p)
        if countZeros == 0:
            return (1, 0)
        return (0, 0)

    def eval(self,state):
        return random.uniform(-0.5,0.5)

    def asHashable(self,state):
        return ''.join([str(v) for v in state[1].flatten()])

    def newGame(self):
        return (1, np.zeros((self.Height,self.Width),dtype = np.int32))




if __name__=='__main__':
    game = Connect4()
    board = np.zeros((game.Height,game.Width),np.int32)
    nextState = (1, board)
    while not game.isEnd(nextState)[0]:
       nextState = random.choice(game.next_states(nextState))
    print(nextState[1])
    print(game.isEnd(nextState))
