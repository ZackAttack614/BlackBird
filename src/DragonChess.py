from FixedMCTS import FixedMCTS
from DynamicMCTS import DynamicMCTS
from GameState import GameState
from proto.state_pb2 import State

import json
import numpy as np

class BoardState(GameState):
    piece_map = { 1: 10, -1: 11, 2: 0, -2: 1, 3: 4, -3: 5, 4: 6, -4: 7, 5: 2, -5: 3, 6: 8, -6: 9 }
    int_to_letter = {
        0: ' ', 1: 'K',  2: 'P',  3: 'N',  4: 'B',  5: 'R',  6: 'Q',
                -1: 'k', -2: 'p', -3: 'n', -4: 'b', -5: 'r', -6: 'q'
    }
    letter_to_int = {
        'K':  1, 'P':  2, 'N':  3, 'B':  4, 'R':  5, 'Q':  6,
        'k': -1, 'p': -2, 'n': -3, 'b': -4, 'r': -5, 'q': -6
    }
    move_to_int = dict()
    int_to_move = dict()
    possible_moves = list()
    GameType = 'DragonChess'
    LegalMoves = 4032

    for square_1 in range(64):
        for square_2 in range(64):
            if square_2 == square_1:
                continue
            move = str(square_1) + ' ' + str(square_2)
            possible_moves.append(move)
            move_to_int[move] = len(possible_moves) - 1
    for key, value in move_to_int.items():
        int_to_move[value] = key

    fen = 'rnbqkbnr/pppppppp/8/8/8/8/3PPP2/4K3 w kq - 0 1'

    def __init__(self):
        self.board = np.zeros((8, 8))
        self.Player = 1
        self.PreviousPlayer = None

        piece_loc, turn, castles, en_passant, halfmove, fullmove = self.fen.split(' ')
        for row_index, row in enumerate(piece_loc.split('/')[::-1]):
            col_index = 0
            for chr in row:
                if chr in self.letter_to_int:
                    self.board[row_index, col_index] = self.letter_to_int[chr]
                elif chr.isnumeric():
                    col_index += int(chr) - 1
                else:
                    raise ValueError(f'FEN position is invalid: {self.fen}')
                col_index += 1

        self.Player = 2 - int(turn == 'w')

        self._white_castle_kingside = 'K' in castles
        self._white_castle_queenside = 'Q' in castles
        self._black_castle_kingside = 'k' in castles
        self._black_castle_queenside = 'q' in castles

    @property
    def Board(self):
        return self.board

    def Copy(self):
        copy = BoardState()
        copy.Player = self.Player
        copy.PreviousPlayer = self.PreviousPlayer
        copy._white_castle_kingside = self._white_castle_kingside
        copy._white_castle_queenside = self._white_castle_queenside
        copy._black_castle_kingside = self._black_castle_kingside
        copy._black_castle_queenside = self._black_castle_queenside

        copy.board = np.copy(self.Board)
        return copy

    def LegalActions(self):
        legal = np.zeros((4032))

        for square_1 in range(64):
            for square_2 in range(64):
                if square_2 == square_1:
                    continue
                col_1, row_1 = square_1 % 8, square_1 // 8
                col_2, row_2 = square_2 % 8, square_2 // 8
                legal[self.move_to_int[f'{square_1} {square_2}']] = int(self._is_legal_move(row_1, col_1, row_2, col_2))

        return legal

    def LegalActionShape(self):
        return np.array([0 for _ in range(4032)], dtype=np.int8)

    def AsInputArray(self):
        state = np.zeros((1, 8, 8, 16), dtype=np.int8)

        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] != 0:
                    state[0, row, col, self.piece_map[self.board[row, col]]] = 1

        state[0, :, :, 12] = int(self._white_castle_kingside)
        state[0, :, :, 13] = int(self._white_castle_queenside)
        state[0, :, :, 14] = int(self._black_castle_kingside)
        state[0, :, :, 15] = int(self._black_castle_queenside)

        return state

    def ApplyAction(self, action):
        move = self.int_to_move[action]
        square_1, square_2 = move.split(' ')
        square_1 = int(square_1)
        square_2 = int(square_2)
        col_1, row_1 = square_1 % 8, square_1 // 8
        col_2, row_2 = square_2 % 8, square_2 // 8

        m = self.Move(row_1, col_1, row_2, col_2)
        if not m:
            raise ValueError('Tried to make an illegal move.')

    def Winner(self, prevAction=None):
        if -1 not in self.board:
            return 1
        elif 1 not in self.board:
            return -1
        else:
            return None

    def EvalToString(self, eval):
        return str(eval)

    def Move(self, loc_row, loc_col, new_row, new_col):
        if self._is_legal_move(loc_row, loc_col, new_row, new_col):
            self.board[new_row, new_col] = self.board[loc_row, loc_col]
            self.board[loc_row, loc_col] = 0
            self.previousPlayer = self.Player
            self.Player = 3 - self.Player
            return True
        else:
            return False
    def __str__(self):
        fboard = np.flip(self.board, axis=0)
        repr = '-----------------\n'
        for row in range(8):
            row_repr = '|'
            for col in range(8):
                row_repr += f'{self.int_to_letter[int(fboard[row, col])]}|'
            repr += row_repr + '\n'
        return repr + '-----------------'

    def __eq__(self, other):
        if other.Player != self.Player:
            return False
        if other._white_castle_kingside != self._white_castle_kingside:
            return False
        if other._white_castle_queenside != self._white_castle_queenside:
            return False
        if other._black_castle_kingside != self._black_castle_kingside:
            return False
        if other._black_castle_queenside != self._black_castle_queenside:
            return False
        return (other.Board == self.Board).all()

    def __hash__(self):
        return "{0}{1}".format(self.Player,str(self)).__hash__()

    def _sanity_check(self, loc_row, loc_col, new_row, new_col):
        if not (-1 < new_row < 8) or not (-1 < new_col < 8): # Moving outside of the board
            return False
        if [new_row, new_col] == [loc_row, loc_col]: # Moving to the square you're already on
            return False
        if self.board[loc_row, loc_col] > 0: # Moving to a square occupied by the same color piece
            if self.board[new_row, new_col] > 0:
                return False
        else:
            if self.board[new_row, new_col] < 0:
                return False
        if self.board[loc_row, loc_col] < 0 and self.Player == 1: # black piece on white's turn
            return False
        elif self.board[loc_row, loc_col] > 0 and self.Player == 2: # white piece on black's turn
            return False
        elif self.board[loc_row, loc_col] == 0: # no piece
            return False
        return True

    def _is_legal_move(self, loc_row, loc_col, new_row, new_col):
        piece_map = {
            1: self._is_legal_move_king,
            2: self._is_legal_move_pawn,
            3: self._is_legal_move_knight,
            4: self._is_legal_move_bishop,
            5: self._is_legal_move_rook,
            6: self._is_legal_move_queen,
        }
        if not self._sanity_check(loc_row, loc_col, new_row, new_col):
            return False

        piece_type = abs(self.board[loc_row, loc_col])
        return piece_map[piece_type](loc_row, loc_col, new_row, new_col)

    def _is_legal_move_pawn(self, loc_row, loc_col, new_row, new_col):
        if not abs(self.board[loc_row, loc_col]) == 2:
            return False
        if self.board[loc_row, loc_col] > 0:
            if loc_row == 1:
                if loc_col == new_col:
                    if new_row == 3 and self.board[3, new_col] == 0 and self.board[2, new_col] == 0:
                        return True
                    elif new_row == 2 and self.board[2, new_col] == 0:
                        return True
                    return False
            if abs(loc_col - new_col) == 1:
                if new_row == loc_row + 1:
                    if self.board[new_row, new_col] < 0:
                        return True
        else:
            if loc_row == 6:
                if loc_col == new_col:
                    if new_row == 4 and self.board[4, new_col] == 0 and self.board[5, new_col] == 0:
                        return True
                    elif new_row == 5 and self.board[5, new_col] == 0:
                        return True
                    return False
            if abs(loc_col - new_col) == 1:
                if new_row == loc_row-1:
                    if self.board[new_row, new_col] > 0:
                        return True
        return False

    def _is_legal_move_rook(self, loc_row, loc_col, new_row, new_col):
        direction = (new_row == loc_row or new_col == loc_col)

        if not direction:
            return False

        no_obstacle = True
        if new_row == loc_row:
            for col in range(min(new_col, loc_col)+1, max(new_col, loc_col)):
                if self.board[new_row, col] != 0:
                    no_obstacle = False
                    break
        else:
            for row in range(min(new_row, loc_row)+1, max(new_row, loc_row)):
                if self.board[row, new_col] != 0:
                    no_obstacle = False
                    break

        return direction and no_obstacle

    def _is_legal_move_bishop(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) != abs(loc_col-new_col):
            return False
        for diag in range(1, abs(loc_row-new_row)):
            if self.board[loc_row+diag*np.sign(new_row-loc_row), loc_col+diag*np.sign(new_col-loc_col)] != 0:
                return False
        return True

    def _is_legal_move_queen(self, loc_row, loc_col, new_row, new_col):
        return self._is_legal_move_bishop(loc_row, loc_col, new_row, new_col) or \
               self._is_legal_move_rook(loc_row, loc_col, new_row, new_col)

    def _is_legal_move_king(self, loc_row, loc_col, new_row, new_col):
        if not abs(self.board[loc_row, loc_col]) == 1:
            return False
        if abs(loc_row-new_row) <= 1 and abs(loc_col-new_col) <= 1:
            return True
        return False

    def _is_legal_move_knight(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) == 1 and abs(loc_col-new_col) == 2:
            return True
        elif abs(loc_row-new_row) == 2 and abs(loc_col-new_col) == 1:
            return True
        return False


if __name__ == "__main__":
    dc = BoardState()
    la = dc.LegalActions()
    for i in range(len(la)):
        if la[i] == 1:
            print(i)