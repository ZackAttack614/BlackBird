import tracemalloc
tracemalloc.start()
from functools import lru_cache
from FixedMCTS import FixedMCTS
from FixedLazyMCTS import FixedLazyMCTS
from DynamicMCTS import DynamicMCTS
from GameState import GameState
from proto.state_pb2 import State

import json
import numpy as np
import time

from memory_profiler import profile
import random

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
    LegalMoves = 4212

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

        self.legal_move_piece_map = {
            1: self._is_legal_move_king,
            2: self._is_legal_move_pawn,
            3: self._is_legal_move_knight,
            4: self._is_legal_move_bishop,
            5: self._is_legal_move_rook,
            6: self._is_legal_move_queen,
        }
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
        legal = np.zeros((4212)) # 4032 ordered pairs of distinct squares + 44*4 = 176 promotions + 4 castles = 4212

        # for square_1 in range(64):
        #     for square_2 in range(64):
        #         if square_2 == square_1:
        #             continue
        #         col_1, row_1 = square_1 % 8, square_1 // 8
        #         col_2, row_2 = square_2 % 8, square_2 // 8
        #         legal[self.move_to_int[f'{square_1} {square_2}']] = int(self._is_legal_move(row_1, col_1, row_2, col_2))

        piece_map = {
            1: self._legal_moves_king,
            2: self._legal_moves_pawn,
            3: self._legal_moves_knight,
            4: self._legal_moves_bishop,
            5: self._legal_moves_rook,
            6: self._legal_moves_queen,
        }

        for row_1 in range(8):
            for col_1 in range(8):
                if np.sign(self.board[row_1, col_1]) == -2*self.Player+3:
                    if abs(self.board[row_1, col_1]) == 1:
                        legal[piece_map[1](self.board, row_1, col_1, -2*self.Player+3, self._white_castle_kingside, self._white_castle_queenside, self._black_castle_kingside, self._black_castle_queenside)]=1
                    else:
                        legal[piece_map[abs(self.board[row_1, col_1])](self.board, row_1, col_1, -2*self.Player+3)] = 1


        return legal

    def NumLegalActions(self):
        numActions = 0
        piece_map = {
            1: self._legal_moves_king,
            2: self._legal_moves_pawn,
            3: self._legal_moves_knight,
            4: self._legal_moves_bishop,
            5: self._legal_moves_rook,
            6: self._legal_moves_queen,
        }

        for row_1 in range(8):
            for col_1 in range(8):
                if np.sign(self.board[row_1, col_1]) == -2*self.Player+3:
                    if abs(self.board[row_1, col_1]) == 1:
                        numActions += len(piece_map[1](self.board, row_1, col_1, -2*self.Player+3, self._white_castle_kingside, self._white_castle_queenside, self._black_castle_kingside, self._black_castle_queenside))
                    else:
                        numActions += len(piece_map[abs(self.board[row_1, col_1])](self.board, row_1, col_1, -2*self.Player+3))
        
        return numActions

    def LegalActionShape(self):
        return np.array([0 for _ in range(4212)], dtype=np.int8)

    def AsInputArray(self):
        state = np.zeros((1, 8, 8, 17), dtype=np.int8)

        for row in range(self.board.shape[0]):
            for col in range(self.board.shape[1]):
                if self.board[row, col] != 0:
                    state[0, row, col, self.piece_map[self.board[row, col]]] = 1

        state[0, :, :, 12] = int(self._white_castle_kingside)
        state[0, :, :, 13] = int(self._white_castle_queenside)
        state[0, :, :, 14] = int(self._black_castle_kingside)
        state[0, :, :, 15] = int(self._black_castle_queenside)
        state[0, :, :, 16] = 1 if self.Player == 1 and self.PreviousPlayer == 1 else 0

        return state

    def ApplyAction(self, action):
        m = False
        if action < 4032:
            move = self.int_to_move[action]
            square_1, square_2 = move.split(' ')
            square_1 = int(square_1)
            square_2 = int(square_2)
            col_1, row_1 = square_1 % 8, square_1 // 8
            col_2, row_2 = square_2 % 8, square_2 // 8

            m = self.Move(row_1, col_1, row_2, col_2)
        else:
            if action >= 4208:
                action -= 4208
                m = self.Move(7*(action//2), 4, 7*(action//2), -4*(action%2)+6, castle=True) 
            else:
                action -= 4032 # 0-175 (first 88 are white promotions)
                promotePiece = (action%4)+3
                if action < 32:
                    m = self.Move(6, action//4, 7, action//4, promote=promotePiece)
                elif action < 60:
                    m = self.Move(6, (action-32)//4, 7, (action-32)//4+1, promote=promotePiece)
                elif action < 88:
                    m = self.Move(6, (action-60)//4+1, 7, (action-60)//4, promote=promotePiece)
                elif action < 120:
                    m = self.Move(1, (action-88)//4, 0, (action-88)//4, promote=-promotePiece)
                elif action < 148:
                    m = self.Move(1, (action-120)//4, 0, (action-120)//4+1, promote=-promotePiece)
                elif action < 176:
                    m = self.Move(1, (action-148)//4+1, 0, (action-148)//4, promote=-promotePiece)

        if not m:
            raise ValueError('Tried to make an illegal move.')

    def Winner(self, prevAction=None):
        if -1 not in self.board:
            return 1
        elif 1 not in self.board:
            return 2
        else:
            return None

    def EvalToString(self, eval):
        return str(eval)

    def Move(self, loc_row, loc_col, new_row, new_col, promote=None, castle=None):
        if promote is not None:
            if self._is_legal_move(loc_row, loc_col, new_row, new_col, promote=True) == 0:
                self.board[new_row, new_col] = promote
            else:
                return False
        elif castle is not None:
            if self._is_legal_move(loc_row, loc_col, new_row, new_col, castle=True) == 0:
                self.board[new_row, new_col] = self.board[loc_row, loc_col]
                self.board[new_row, new_col//2+2] = self.board[new_row, new_col*7//4-3] # rook
                self.board[new_row, new_col*7//4-3] = 0
            else:
                return False
        elif self._is_legal_move(loc_row, loc_col, new_row, new_col):
            self.board[new_row, new_col] = self.board[loc_row, loc_col]
        else:
            return False

        # general things
        self.board[loc_row, loc_col] = 0
        if self.PreviousPlayer == 1 and self.Player == 1:
            self.Player = 2
            self.PreviousPlayer = 1
        else:
            self.PreviousPlayer = self.Player
            self.Player = 1

        if self.board[0, 4] != 1:
            self._white_castle_kingside = False
            self._white_castle_queenside = False
        elif self.board[0, 7] != 5:
            self._white_castle_kingside = False
        if self.board[0, 0] != 5:
            self._white_castle_queenside = False
        if self.board[7, 4] != -1:
            self._black_castle_kingside = False
            self._black_castle_queenside = False
        elif self.board[7, 7] != -5:
            self._black_castle_kingside = False
        if self.board[7, 0] != -5:
            self._black_castle_queenside = False

        return True

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
        if (square := self.board[loc_row, loc_col]) == 0: # no piece
            return False
        elif square < 0 and self.Player == 1: # black piece on white's turn
            return False
        elif square > 0 and self.Player == 2: # white piece on black's turn
            return False
        if abs(loc_row-new_row) != abs(loc_col-new_col) and \
            ((abs(loc_row-new_row) >= 3 and abs(loc_col-new_col) >= 1) or \
             (abs(loc_col-new_col) >= 3 and abs(loc_row-new_row) >= 1)):
            return False
        if square > 0: # Moving to a square occupied by the same color piece
            if self.board[new_row, new_col] > 0:
                return False
        else:
            if self.board[new_row, new_col] < 0:
                return False
        if not (-1 < new_row < 8) or not (-1 < new_col < 8): # Moving outside of the board
            return False
        if new_row == loc_row and new_col == loc_col: # Moving to the square you're already on
            return False
        return True

    def _is_legal_move(self, loc_row, loc_col, new_row, new_col, promote=None, castle=None):
        if not self._sanity_check(loc_row, loc_col, new_row, new_col):
            return False

        piece_type = abs(self.board[loc_row, loc_col])
        if promote is not None and piece_type != 2:
            return False
        if castle is not None and piece_type != 1:
            return False
        return self.legal_move_piece_map[piece_type](loc_row, loc_col, new_row, new_col)

    def _is_legal_move_pawn(self, loc_row, loc_col, new_row, new_col):
        if self.board[loc_row, loc_col] > 0:
            if loc_row == 6 and loc_col == new_col and new_row == 7 and self.board[7, new_col] == 0:
                return 0
            elif loc_row == 6 and abs(loc_col - new_col) == 1 and new_row == 7 and self.board[7, new_col] < 0:
                return 0
            elif loc_row == 1:
                if loc_col == new_col:
                    if new_row == 3 and self.board[3, new_col] == 0 and self.board[2, new_col] == 0:
                        return True
                    elif new_row == 2 and self.board[2, new_col] == 0:
                        return True
                    return False
            elif loc_row > 1 and loc_row < 6:
                if loc_col == new_col and self.board[new_row, new_col] == 0 and new_row == loc_row+1:
                    return True
            if abs(loc_col - new_col) == 1:
                if new_row == loc_row + 1:
                    if self.board[new_row, new_col] < 0:
                        return True
        else:
            if loc_row == 1 and loc_col == new_col and new_row == 0 and self.board[0, new_col] == 0:
                return 0
            elif loc_row == 1 and abs(loc_col - new_col) == 1 and new_row == 0 and self.board[0, new_col] > 0:
                return 0
            elif loc_row == 6:
                if loc_col == new_col:
                    if new_row == 4 and self.board[4, new_col] == 0 and self.board[5, new_col] == 0:
                        return True
                    elif new_row == 5 and self.board[5, new_col] == 0:
                        return True
                    return False
            elif loc_row > 1 and loc_row < 6:
                if loc_col == new_col and self.board[new_row, new_col] == 0 and new_row == loc_row-1:
                    return True
            if abs(loc_col - new_col) == 1:
                if new_row == loc_row-1:
                    if self.board[new_row, new_col] > 0:
                        return True
        return False

    @staticmethod
    def _legal_moves_pawn(board, loc_row, loc_col, color):
        result = []
        if color == 1:
            if loc_row != 6:
                if loc_col != 0: #left capture
                    if np.sign(board[loc_row+1, loc_col-1]) == -1:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col + 7
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                if loc_col != 7: #right capture
                    if np.sign(board[loc_row+1, loc_col+1]) == -1:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col + 9
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                if np.sign(board[loc_row+1, loc_col]) == 0: # move up one
                    square_1 = 8*loc_row + loc_col
                    square_2 = 8*loc_row + loc_col + 8
                    result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                    if loc_row == 1 and np.sign(board[loc_row+2, loc_col]) == 0: # move up two
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col + 16
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
            else:
                DEFact = 4032
                if loc_col != 0: #left capture
                    if np.sign(board[loc_row+1, loc_col-1]) == -1:
                        for promotion in range(4):
                            result.append(DEFact+60+4*(loc_col-1)+promotion)
                if loc_col != 7: #right capture
                    if np.sign(board[loc_row+1, loc_col+1]) == -1:
                        for promotion in range(4):
                            result.append(DEFact+32+4*loc_col+promotion)
                if np.sign(board[loc_row+1, loc_col]) == 0: # move up one
                    for promotion in range(4):
                        result.append(DEFact+4*loc_col+promotion)
        else:
            if loc_row != 1:
                if loc_col != 0: #left capture
                    if np.sign(board[loc_row-1, loc_col-1]) == 1:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col - 9
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                if loc_col != 7: #right capture
                    if np.sign(board[loc_row-1, loc_col+1]) == 1:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col - 7
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                if np.sign(board[loc_row-1, loc_col]) == 0: # move up one
                    square_1 = 8*loc_row + loc_col
                    square_2 = 8*loc_row + loc_col - 8
                    result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                    if loc_row == 6 and np.sign(board[loc_row-2, loc_col]) == 0: # move up two
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*loc_row + loc_col - 16
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
            else:
                DEFact = 4120
                if loc_col != 0: #left capture
                    if np.sign(board[loc_row-1, loc_col-1]) == 1:
                        for promotion in range(4):
                            result.append(DEFact+60+4*(loc_col-1)+promotion)
                if loc_col != 7: #right capture
                    if np.sign(board[loc_row-1, loc_col+1]) == 1:
                        for promotion in range(4):
                            result.append(DEFact+32+4*loc_col+promotion)
                if np.sign(board[loc_row-1, loc_col]) == 0: # move up one
                    for promotion in range(4):
                        result.append(DEFact+4*loc_col+promotion)
        return result
                    
        

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

    @staticmethod
    def _legal_moves_rook(board, loc_row, loc_col, color):
        result = []
        for dir in range(4):
            theta = (dir*90)*np.pi/180
            count = 1
            while True:
                adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                new_row = loc_row + count*adjCos
                new_col = loc_col + count*adjSin
                if (-1 < new_row < 8) and (-1 < new_col < 8):
                    if np.sign(board[new_row, new_col]) == 0:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                    elif np.sign(board[new_row, new_col]) != color:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                        break
                    else:
                        break
                else:
                    break
                count += 1
        return result

    def _is_legal_move_bishop(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) != abs(loc_col-new_col):
            return False
        for diag in range(1, abs(loc_row-new_row)):
            if self.board[loc_row+diag*np.sign(new_row-loc_row), loc_col+diag*np.sign(new_col-loc_col)] != 0:
                return False
        return True

    @staticmethod
    def _legal_moves_bishop(board, loc_row, loc_col, color):
        result = []
        for dir in range(4):
            theta = (dir*90 + 45)*np.pi/180
            count = 1
            while True:
                adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                new_row = loc_row + count*adjCos
                new_col = loc_col + count*adjSin
                if (-1 < new_row < 8) and (-1 < new_col < 8):
                    if np.sign(board[new_row, new_col]) == 0:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                    elif np.sign(board[new_row, new_col]) != color:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                        break
                    else:
                        break
                else:
                    break
                count += 1
        return result

    def _is_legal_move_queen(self, loc_row, loc_col, new_row, new_col):
        return self._is_legal_move_bishop(loc_row, loc_col, new_row, new_col) or \
               self._is_legal_move_rook(loc_row, loc_col, new_row, new_col)

    @staticmethod
    def _legal_moves_queen(board, loc_row, loc_col, color):
        result = []
        for dir in range(8):
            theta = (dir*45)*np.pi/180
            count = 1
            while True:
                adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                new_row = loc_row + count*adjCos
                new_col = loc_col + count*adjSin
                if (-1 < new_row < 8) and (-1 < new_col < 8):
                    if np.sign(board[new_row, new_col]) == 0:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                    elif np.sign(board[new_row, new_col]) != color:
                        square_1 = 8*loc_row + loc_col
                        square_2 = 8*new_row + new_col
                        result.append(63*square_1+square_2+int(square_2 < square_1)-1)
                        break
                    else:
                        break
                else:
                    break
                count += 1
        return result

    def _is_legal_move_king(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) <= 1 and abs(loc_col-new_col) <= 1:
            return True
        elif loc_row == 0 and new_col == 6 and self._white_castle_kingside and (self.board[0, 5:7] == 0).all():
            return 0
        elif loc_row == 0 and new_col == 2 and self._white_castle_queenside and (self.board[0, 1:4] == 0).all():
            return 0
        elif loc_row == 7 and new_col == 6 and self._black_castle_kingside and (self.board[7, 5:7] == 0).all():
            return 0
        elif loc_row == 7 and new_col == 2 and self._black_castle_queenside and (self.board[7, 1:4] == 0).all():
            return 0
        return False

    @staticmethod
    def _legal_moves_king(board, loc_row, loc_col, color, white_castle_kingside, white_castle_queenside, black_castle_kingside, black_castle_queenside):
        result = []
        for drow in range(-1, 2):
            for dcol in range(-1, 2):
                new_row = loc_row + drow
                new_col = loc_col + dcol
                if (-1 < new_row < 8) and (-1 < new_col < 8) \
                and (drow != 0 or dcol != 0) \
                and np.sign(board[new_row, new_col]) != color:
                    square_1 = 8*loc_row + loc_col
                    square_2 = 8*new_row + new_col
                    result.append(63*square_1+square_2+int(square_2 < square_1)-1)
        if color == 1:
            if loc_row == 0 and loc_col == 4 and white_castle_kingside and (board[0, 5:7] == 0).all():
                result.append(4208)
            if loc_row == 0 and loc_col == 4 and white_castle_queenside and (board[0, 1:4] == 0).all():
                result.append(4209)
        else:
            if loc_row == 7 and loc_col == 4 and black_castle_kingside and (board[7, 5:7] == 0).all():
                result.append(4210)
            if loc_row == 7 and loc_col == 4 and black_castle_queenside and (board[7, 1:4] == 0).all():
                result.append(4211)
        return result

    def _is_legal_move_knight(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) == 1 and abs(loc_col-new_col) == 2:
            return True
        elif abs(loc_row-new_row) == 2 and abs(loc_col-new_col) == 1:
            return True
        return False

    @staticmethod
    def _legal_moves_knight(board, loc_row, loc_col, color):
        result = []
        for drow in (-2, -1, 1, 2):
            new_row = loc_row + drow
            for dcol in (3-abs(drow), -3+abs(drow)):
                new_col = loc_col + dcol
                if (-1 < new_row < 8) and (-1 < new_col < 8) \
                and np.sign(board[new_row, new_col]) != color:
                    square_1 = 8*loc_row + loc_col
                    square_2 = 8*new_row + new_col
                    result.append(63*square_1+square_2+int(square_2 < square_1)-1)
        return result

def main():
    import cProfile, pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    params = {'maxDepth' : 10, 'explorationRate' : 0.05, 'playLimit' : 1000}
    player = FixedLazyMCTS(**params)

    state = BoardState()
    while state.Winner() is None:
        print(state)
        print('To move: {}'.format(state.Player))
        state, v, p = player.FindMove(state, temp=player.ExplorationRate)
        print('Value: {}'.format(v))
        print('Selection Probabilities: {}'.format(p))
        print('Child Values: {}'.format(player.Root.ChildWinRates()))
        print('Child Exploration Rates: {}'.format(player.Root.ChildPlays()))
        print()
        player.MoveRoot(state)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile.log')
    print(state)
    print(state.Winner())

if __name__ == "__main__":
    main()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)
    # import cProfile, pstats
    
    # profiler = cProfile.Profile()
    # profiler.enable()

    # now = time.time()
    # for game in range(100):
    #     # print(f'Playing game {game+1}')
    #     dc = BoardState()
    #     # print(dc)
    #     while dc.Winner() == None:
    #         # choice = input("Enter a move: ")
    #         actions = dc.LegalActions()
    #         choice = np.random.choice(np.where(actions == 1)[0])
    #         try:
    #             # if ' ' in choice:
    #             #     choice = dc.move_to_int[choice]
    #             # print(choice)
    #             # if choice < 4032:
    #             #     print(dc.int_to_move[choice])
    #             # elif choice < 4208:
    #             #     print('PROMOTION')
    #             # else:
    #             #     print('CASTLE')
    #             # input()
    #             dc.ApplyAction(choice)
    #         except ValueError:
    #             print("The move you entered was illegal.")
    #         # print(dc)
    #     # print(dc.Winner())

    # print(time.time()-now)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.dump_stats('profile.log')