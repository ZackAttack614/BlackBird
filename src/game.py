import numpy as np

class game:
    def __init__(self, dim=3):
        self.dim = dim  # Dimensionality of the square board
        self.reset()
        
    def reset(self):
        """ Return the game to its starting state.
        """

        self.player = 1
        self.x_state = np.zeros((self.dim, self.dim))
        self.o_state = np.zeros((self.dim, self.dim))
        self.board = np.zeros((1, self.dim, self.dim, 2))        
        
    def getResult(self):
        """ Return the result of a terminal game state.
            1  -> X won
            0  -> Draw
            -1 -> O won
        """

        if self.dim in np.hstack([
            np.sum(self.x_state, axis=0),
            np.sum(self.x_state, axis=1),
            np.sum(self.x_state[i, i] for i in range(self.dim)),
            np.sum(self.x_state[i, self.dim-i-1] for i in range(self.dim))
        ]):
            return 1
        elif self.dim in np.hstack([
            np.sum(self.o_state, axis=0),
            np.sum(self.o_state, axis=1),
            np.sum(self.o_state[i, i] for i in range(self.dim)),
            np.sum(self.o_state[i, self.dim-i-1] for i in range(self.dim))
        ]):
            return -1
        else:
            return 0
        
    def isGameOver(self):
        """ Determine whether the current game state is a terminal position.
        """

        if self.getResult() != 0:
            self.gameover = True
            return True
        elif 0 not in (self.board[0,:,:,0] + self.board[0,:,:,1]):
            self.gameover = True
            return True
        else:
            return False
        
    def move(self, position):
        """ Apply a move to the current game state.
        """
        if self.player == 1:
            self.x_state[position] = 1
        elif self.player == -1:
            self.o_state[position] = 1
        self.board[0,:,:,0] = self.x_state
        self.board[0,:,:,1] = self.o_state
        
        self.player *= -1
        
    def getLegalMoves(self):
        """ Return a list of all moves that are legal in the current game state.
        """
        legal_moves = []
        
        for i in range(self.dim):
            for j in range(self.dim):
                if self.board[0, i, j, 0] == 0 and self.board[0, i, j, 1] == 0:
                    legal_moves.append((i, j))
                    
        return legal_moves
    
    def dumpBoard(self):
        """ Display the board state to the user
        """
        print(str(self))
        print('\n')

    def __str__(self):
        s = ''
        for i in range(self.dim):
            next_line = [' ']*self.dim
            for j in range(self.dim):
                if self.board[0,i,j,0] == 1:
                    next_line[j]='X'
                elif self.board[0,i,j,1] == 1:
                    next_line[j]='O'
            s += '|'.join(next_line) + '\n'
            if i != self.dim-1:
                s += '-'*(2*self.dim-1) + '\n'
        return s
