import tracemalloc
tracemalloc.start()
from functools import lru_cache
from FixedMCTS import FixedMCTS
from FixedLazyMCTS import FixedLazyMCTS
from DynamicMCTS import DynamicMCTS
from GameState import GameState
# from proto.state_pb2 import State

import json
import numpy as np
import time

import random

class BoardState(GameState):
    piece_map = { 5: 10, -5: 11, 2: 4, 7: 4, -2: 5, -7: 5, 3: 6, 6: 6, -3: 7, -6: 7, 1: 2, 8: 2, -1: 3, -8: 3, 4: 8, -4: 9 }
    for i in range(9, 17):
        piece_map[i] = 0
        piece_map[i-25] = 1
    for i in range(17, 25):
        piece_map[i] = 4
        piece_map[i-41] = 5
    for i in range(25, 33):
        piece_map[i] = 6
        piece_map[i-57] = 7
    for i in range(33, 41):
        piece_map[i] = 2
        piece_map[i-73] = 3
    for i in range(41, 49):
        piece_map[i] = 8
        piece_map[i-89] = 9
    GameType = 'DragonChess'
    LegalMoves = 802

    def __init__(self):
        self.board = np.array([[0, 0, 0, 0, 5, 0, 0, 0], [0, 0, 0, 12, 13, 14, 0, 0], [0]*8, [0]*8, [0]*8, [0]*8, [-16, -15, -14, -13, -12, -11, -10, -9], [-1, -2, -3, -4, -5, -6, -7, -8]], dtype=np.int8)
        self.Player = 1
        self._white_castle_kingside = False
        self._white_castle_queenside = False
        self._black_castle_kingside = True
        self._black_castle_queenside = True
        self.PreviousPlayer = None
        self.playedMoves = 0
        self.LastAction = None

        self.legal_move_piece_map = {
            1: self._is_legal_move_rook,
            8: self._is_legal_move_rook,
            2: self._is_legal_move_knight,
            7: self._is_legal_move_knight,
            3: self._is_legal_move_bishop,
            6: self._is_legal_move_bishop,
            4: self._is_legal_move_queen,
            5: self._is_legal_move_king,
        }
        for i in range(9, 17):
            self.legal_move_piece_map[i] = self._is_legal_move_pawn
        for i in range(17, 25):
            self.legal_move_piece_map[i] = self._is_legal_move_knight
        for i in range(25, 33):
            self.legal_move_piece_map[i] = self._is_legal_move_bishop
        for i in range(33, 41):
            self.legal_move_piece_map[i] = self._is_legal_move_rook
        for i in range(41, 49):
            self.legal_move_piece_map[i] = self._is_legal_move_queen

    @property
    def Board(self):
        return self.board

    def Copy(self):
        copy = BoardState()
        copy.Player = self.Player
        copy.LastAction = self.LastAction
        copy.PreviousPlayer = self.PreviousPlayer
        copy._white_castle_kingside = self._white_castle_kingside
        copy._white_castle_queenside = self._white_castle_queenside
        copy._black_castle_kingside = self._black_castle_kingside
        copy._black_castle_queenside = self._black_castle_queenside

        copy.board = np.copy(self.Board)
        copy.playedMoves = self.playedMoves
        return copy

    def LegalActions(self):
        legal = np.zeros((802)) # 28*2 (rooks) + 8*2 (knights) + 28*2 (bishops) + 56 (queen) + 8 (king) + 64*8 (pawns) + 96 promotions + 2 castles
        # 0-55, 56-71, 72-127, 128-183, 184-191, 192-703, 704-799
        ############################################## Look for mate
        # oppKing = np.where(self.board == 10*self.Player - 15)
        # mateFrom = []
        # if len(oppKing[0]) >= 1:
        #     oppKingRow = oppKing[0][0]
        #     oppKingCol = oppKing[1][0]
        #     for row in range(8):
        #         for col in range(8):
        #             if self._is_legal_move(row, col, oppKingRow, oppKingCol) or self._is_legal_move(row, col, oppKingRow, oppKingCol, promote=True) == 2:
        #                 mateFrom.append((row, col))
        # if len(mateFrom) > 0:
        #     possMoves = self._legal_moves_any(self.board, mateFrom[0][0], mateFrom[0][1], -2*self.Player+3, self._white_castle_kingside, self._white_castle_queenside, self._black_castle_kingside, self._black_castle_queenside)
        #     for move in possMoves:
        #         newGame = self.Copy()
        #         newGame.ApplyAction(move)
        #         if newGame.Winner() > 0:
        #             legal[move] = 1
        #             return legal
        #     raise Exception("Failed to find mating move.")
        ##########################################################
        for row_1 in range(8):
            for col_1 in range(8):
                if np.sign(self.board[row_1, col_1]) == -2*self.Player+3:
                    legal[self._legal_moves_any(self.board, row_1, col_1, -2*self.Player+3, self._white_castle_kingside, self._white_castle_queenside, self._black_castle_kingside, self._black_castle_queenside)]=1

        return legal

    def NumLegalActions(self):
        numActions = 0
        ################## Mating move exists
        # oppKing = np.where(self.board == 10*self.Player - 15)
        # mateFrom = []
        # if len(oppKing[0]) >= 1:
        #     oppKingRow = oppKing[0][0] # board is flipped for white perspective
        #     oppKingCol = oppKing[1][0]
        #     for row in range(8):
        #         for col in range(8):
        #             if self._is_legal_move(row, col, oppKingRow, oppKingCol) or self._is_legal_move(row, col, oppKingRow, oppKingCol, promote=True) == 2:
        #                 mateFrom.append((row, col))
        # if len(mateFrom) > 0:
        #     return 1
        ################# 
        for row_1 in range(8):
            for col_1 in range(8):
                if np.sign(self.board[row_1, col_1]) == -2*self.Player+3:
                    numActions += len(self._legal_moves_any(self.board, row_1, col_1, -2*self.Player+3, self._white_castle_kingside, self._white_castle_queenside, self._black_castle_kingside, self._black_castle_queenside))
        
        return numActions

    def LegalActionShape(self):
        return np.array([0 for _ in range(802)], dtype=np.int8)

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
        if action < 28: #rook
            piece = np.where(self.board==(-2*self.Player+3))
            theta = (action//7)*np.pi/2
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(action%7+1)*adjSin, piece[1][0]+(action%7+1)*adjCos)
        elif action < 56: #rook
            piece = np.where(self.board==(-2*self.Player+3)*8)
            theta = ((action-28)//7)*np.pi/2
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(action%7+1)*adjSin, piece[1][0]+(action%7+1)*adjCos)
        elif action < 64: #knight
            piece = np.where(self.board==(-2*self.Player+3)*2)
            ydists = {0: 2, 1: 1, 2: -1, 3: -2, 4: -2, 5: -1, 6: 1, 7: 2}
            xdists = {0: 1, 1: 2, 2: 2, 3: 1, 4: -1, 5: -2, 6: -2, 7: -1}
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+xdists[action-56], piece[1][0]+ydists[action-56])
        elif action < 72: #knight
            piece = np.where(self.board==(-2*self.Player+3)*7)
            ydists = {0: 2, 1: 1, 2: -1, 3: -2, 4: -2, 5: -1, 6: 1, 7: 2}
            xdists = {0: 1, 1: 2, 2: 2, 3: 1, 4: -1, 5: -2, 6: -2, 7: -1}
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+xdists[action-64], piece[1][0]+ydists[action-64])
        elif action < 100: #bishop
            piece = np.where(self.board==(-2*self.Player+3)*3)
            theta = (90*((action-72)//7)+45)*np.pi/180
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+((action-72)%7+1)*adjSin, piece[1][0]+((action-72)%7+1)*adjCos)
        elif action < 128: #bishop
            piece = np.where(self.board==(-2*self.Player+3)*6)
            theta = (90*((action-100)//7)+45)*np.pi/180
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+((action-100)%7+1)*adjSin, piece[1][0]+((action-100)%7+1)*adjCos)
        elif action < 184: #queen
            piece = np.where(self.board==(-2*self.Player+3)*4)
            theta = ((action-128)//7)*np.pi/4
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+((action-128)%7+1)*adjSin, piece[1][0]+((action-128)%7+1)*adjCos)
        elif action < 192: #king
            piece = np.where(self.board==(-2*self.Player+3)*5)
            theta = ((action-184)%8)*np.pi/4
            adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
            adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
            m = self.Move(piece[0][0], piece[1][0], piece[0][0]+adjSin, piece[1][0]+adjCos)
        elif action < 704:
            piece = None
            for i in range((action-192)//64+9, (action-192)//64+49, 8):
                piece = np.where(self.board==(-2*self.Player+3)*i)
                if len(piece[0]) > 0:
                    break
            if action % 8 == 7:
                ydists = {0: 2, 1: 1, 2: -1, 3: -2, 4: -2, 5: -1, 6: 1, 7: 2}
                xdists = {0: 1, 1: 2, 2: 2, 3: 1, 4: -1, 5: -2, 6: -2, 7: -1}
                m = self.Move(piece[0][0], piece[1][0], piece[0][0]+xdists[(action%64)//8], piece[1][0]+ydists[(action%64)//8])
            else:
                theta = ((action%64)//8)*np.pi/4
                adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(action%8+1)*adjSin, piece[1][0]+(action%8+1)*adjCos)
        else:
            if action >= 800:
                action -= 800
                m = self.Move(7*(action//2), 4, 7*(action//2), -4*(action%2)+6, castle=True) 
                action += 800
            else:
                action -= 704
                promotePiece = 17+(action//12)+8*(action%4)
                piece = np.where(self.board==(-2*self.Player+3)*(action//12+9))
                if action%12 < 4:
                    m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(-2*self.Player+3), piece[1][0]-1, promote=(-2*self.Player+3)*promotePiece)
                elif action%12 < 8:
                    m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(-2*self.Player+3), piece[1][0], promote=(-2*self.Player+3)*promotePiece)
                else:
                    m = self.Move(piece[0][0], piece[1][0], piece[0][0]+(-2*self.Player+3), piece[1][0]+1, promote=(-2*self.Player+3)*promotePiece)
                action += 704

        if not m:
            print(self, action, self.Player, self.PreviousPlayer)
            raise ValueError('Tried to make an illegal move.')

        self.playedMoves += 1
        self.LastAction = action

    def Winner(self, prevAction=None):
        if -5 not in self.board:
            return 1
        elif 5 not in self.board:
            return 2
        elif self.playedMoves >= 80:
            return 0
        else:
            return -1

    def EvalToString(self, eval):
        return str(eval)

    def Move(self, loc_row, loc_col, new_row, new_col, promote=None, castle=None):
        if promote is not None:
            if self._is_legal_move(loc_row, loc_col, new_row, new_col, promote=True) == 2:
                self.board[new_row, new_col] = promote
            else:
                return False
        elif castle is not None:
            if self._is_legal_move(loc_row, loc_col, new_row, new_col, castle=True) == 2:
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
                piece = abs(fboard[row, col])
                nextChar = ' '
                if piece != 0:
                    if piece == 1 or piece == 8 or piece in range(33, 41):
                        nextChar = 'r'
                    elif piece == 2 or piece == 7 or piece in range(17, 25):
                        nextChar = 'n'
                    elif piece == 3 or piece == 6 or piece in range(25, 33):
                        nextChar = 'b'
                    elif piece == 4 or piece in range(41, 49):
                        nextChar = 'q'
                    elif piece == 5:
                        nextChar = 'k'
                    elif piece in range(9, 17):
                        nextChar = 'p'
                row_repr += (nextChar if np.sign(fboard[row, col]) <= 0 else nextChar.upper()) + '|'
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
    
    def eq(self, other):
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
        if promote is not None and piece_type not in range(9, 17):
            return False
        if castle is not None and piece_type != 5:
            return False
        return self.legal_move_piece_map[piece_type](loc_row, loc_col, new_row, new_col)

    def _is_legal_move_pawn(self, loc_row, loc_col, new_row, new_col):
        if self.board[loc_row, loc_col] > 0:
            if loc_row == 6 and loc_col == new_col and new_row == 7 and self.board[7, new_col] == 0:
                return 2
            elif loc_row == 6 and abs(loc_col - new_col) == 1 and new_row == 7 and self.board[7, new_col] < 0:
                return 2
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
                return 2
            elif loc_row == 1 and abs(loc_col - new_col) == 1 and new_row == 0 and self.board[0, new_col] > 0:
                return 2
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
        if abs(loc_row-new_row) <= 1 and abs(loc_col-new_col) <= 1:
            return True
        elif loc_row == 0 and new_col == 6 and new_row == 0 and self._white_castle_kingside and (self.board[0, 5:7] == 0).all():
            return 2
        elif loc_row == 0 and new_col == 2 and new_row == 0 and self._white_castle_queenside and (self.board[0, 1:4] == 0).all():
            return 2
        elif loc_row == 7 and new_col == 6 and new_row == 7 and self._black_castle_kingside and (self.board[7, 5:7] == 0).all():
            return 2
        elif loc_row == 7 and new_col == 2 and new_row == 7 and self._black_castle_queenside and (self.board[7, 1:4] == 0).all():
            return 2
        return False

    def _is_legal_move_knight(self, loc_row, loc_col, new_row, new_col):
        if abs(loc_row-new_row) == 1 and abs(loc_col-new_col) == 2:
            return True
        elif abs(loc_row-new_row) == 2 and abs(loc_col-new_col) == 1:
            return True
        return False

    @staticmethod
    def _legal_moves_any(board, loc_row, loc_col, color, white_castle_kingside, white_castle_queenside, black_castle_kingside, black_castle_queenside):
        piece = abs(board[loc_row, loc_col])
        if piece == 1 or piece == 8: #rook
            result = []
            for dir in range(4):
                theta = (dir*90)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(28*((piece-1)//7)+7*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(28*((piece-1)//7)+7*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result
        elif piece == 2 or piece == 7: #knight
            result = []
            for drow in (-2, -1, 1, 2):
                new_row = loc_row + drow
                for dcol in (3-abs(drow), -3+abs(drow)):
                    new_col = loc_col + dcol
                    if (-1 < new_row < 8) and (-1 < new_col < 8) \
                    and np.sign(board[new_row, new_col]) != color:
                        result.append(56+8*((piece-2)//5)+np.sign(drow)*(-dcol+2+(np.sign(dcol)-1)//2) + -7*((np.sign(drow)-1)//2))
            return result
        elif piece == 3 or piece == 6: #bishop
            result = []
            for dir in range(4):
                theta = (dir*90 + 45)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(72+28*((piece-3)//3)+7*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(72+28*((piece-3)//3)+7*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result
        elif piece == 4: #queen
            result = []
            for dir in range(8):
                theta = (dir*45)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(128+7*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(128+7*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result
        elif piece == 5: #king
            result = []
            for dir in range(8):
                theta = (dir*45)*np.pi/180
                adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                new_row = loc_row + adjSin
                new_col = loc_col + adjCos
                if (-1 < new_row < 8) and (-1 < new_col < 8) and np.sign(board[new_row, new_col]) != color:
                    result.append(184+dir)
            if color == 1:
                if loc_row == 0 and loc_col == 4 and white_castle_kingside and (board[0, 5:7] == 0).all():
                    result.append(800)
                if loc_row == 0 and loc_col == 4 and white_castle_queenside and (board[0, 1:4] == 0).all():
                    result.append(801)
            else:
                if loc_row == 7 and loc_col == 4 and black_castle_kingside and (board[7, 5:7] == 0).all():
                    result.append(800)
                if loc_row == 7 and loc_col == 4 and black_castle_queenside and (board[7, 1:4] == 0).all():
                    result.append(801)
            return result
        elif piece in range(9, 17): #pawn
            result = []
            DEFact = 192+64*(piece-9)
            if color == 1:
                if loc_row != 6:
                    if loc_col != 0: #left capture
                        if np.sign(board[loc_row+1, loc_col-1]) == -1:
                            result.append(DEFact+24)
                    if loc_col != 7: #right capture
                        if np.sign(board[loc_row+1, loc_col+1]) == -1:
                            result.append(DEFact+8)
                    if np.sign(board[loc_row+1, loc_col]) == 0: # move up one
                        result.append(DEFact+16)
                        if loc_row == 1 and np.sign(board[loc_row+2, loc_col]) == 0: # move up two
                            result.append(DEFact+17)
                else:
                    PromDEFact = 704 + 12*(piece-9)
                    if loc_col != 0: #left capture
                        if np.sign(board[loc_row+1, loc_col-1]) == -1:
                            for promotion in range(4):
                                result.append(PromDEFact+promotion)
                    if loc_col != 7: #right capture
                        if np.sign(board[loc_row+1, loc_col+1]) == -1:
                            for promotion in range(4):
                                result.append(PromDEFact+8+promotion)
                    if np.sign(board[loc_row+1, loc_col]) == 0: # move up one
                        for promotion in range(4):
                            result.append(PromDEFact+4+promotion)
            else:
                if loc_row != 1:
                    if loc_col != 0: #left capture
                        if np.sign(board[loc_row-1, loc_col-1]) == 1:
                            result.append(DEFact+40)
                    if loc_col != 7: #right capture
                        if np.sign(board[loc_row-1, loc_col+1]) == 1:
                            result.append(DEFact+56)
                    if np.sign(board[loc_row-1, loc_col]) == 0: # move up one
                        result.append(DEFact+48)
                        if loc_row == 6 and np.sign(board[loc_row-2, loc_col]) == 0: # move up two
                            result.append(DEFact+49)
                else:
                    PromDEFact = 704 + 12*(piece-9)
                    if loc_col != 0: #left capture
                        if np.sign(board[loc_row-1, loc_col-1]) == 1:
                            for promotion in range(4):
                                result.append(PromDEFact+promotion)
                    if loc_col != 7: #right capture
                        if np.sign(board[loc_row-1, loc_col+1]) == 1:
                            for promotion in range(4):
                                result.append(PromDEFact+8+promotion)
                    if np.sign(board[loc_row-1, loc_col]) == 0: # move up one
                        for promotion in range(4):
                            result.append(PromDEFact+4+promotion)
            return result
        elif piece in range(17, 25): #knight
            result = []
            DEFact = 192+64*(piece-17)
            for drow in (-2, -1, 1, 2):
                new_row = loc_row + drow
                for dcol in (3-abs(drow), -3+abs(drow)):
                    new_col = loc_col + dcol
                    if (-1 < new_row < 8) and (-1 < new_col < 8) \
                    and np.sign(board[new_row, new_col]) != color:
                        result.append(DEFact+7+8*(np.sign(drow)*(-dcol+2+(np.sign(dcol)-1)//2) + -7*((np.sign(drow)-1)//2)))
            return result
        elif piece in range(25, 33): #bishop
            result = []
            DEFact = 192+64*(piece-25)
            for dir in range(4):
                theta = (dir*90 + 45)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(DEFact+8+16*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(DEFact+8+16*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result
        elif piece in range(33, 41): #rook
            result = []
            DEFact = 192+64*(piece-33)
            for dir in range(4):
                theta = (dir*90)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(DEFact+16*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(DEFact+16*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result
        elif piece in range(41, 49): #queen
            result = []
            DEFact = 192+64*(piece-41)
            for dir in range(8):
                theta = (dir*45)*np.pi/180
                count = 1
                while True:
                    adjCos = 0 if np.abs(c:=np.cos(theta)) < 1e-8 else int(np.sign(c))
                    adjSin = 0 if np.abs(s:=np.sin(theta)) < 1e-8 else int(np.sign(s))
                    new_row = loc_row + count*adjSin
                    new_col = loc_col + count*adjCos
                    if (-1 < new_row < 8) and (-1 < new_col < 8):
                        if np.sign(board[new_row, new_col]) == 0:
                            result.append(DEFact+8*dir+count-1)
                        elif np.sign(board[new_row, new_col]) != color:
                            result.append(DEFact+8*dir+count-1)
                            break
                        else:
                            break
                    else:
                        break
                    count += 1
            return result

def main():
    # state = BoardState()
    # state.ApplyAction(783)
    # state.ApplyAction(711)
    # state.ApplyAction(3320)
    # state.ApplyAction(839)
    # state.ApplyAction(264)
    # state.ApplyAction(3184)
    # state.ApplyAction(840)
    # state.ApplyAction(1223)
    # print(state.LegalActions())
    # state.ApplyAction(3953)
    # print(np.where(state.LegalActions() == 1)[0])
    # print(state)
    # exit()
    import cProfile, pstats
    
    profiler = cProfile.Profile()
    profiler.enable()
    params = {'maxDepth' : 10, 'explorationRate' : 0.05, 'playLimit' : 1000}
    player = FixedLazyMCTS(**params)

    state = BoardState()
    start = time.time()
    while state.Winner() == -1:
        print(state)
        print('To move: {}'.format(state.Player))
        print('Number of Legal Moves: {}'.format(state.NumLegalActions()))
        state, v, p = player.FindMove(state, temp=player.ExplorationRate)
        print('Value: {}'.format(v))
        # print('Selection Probabilities: {}'.format(p))
        print('Child Values: {}'.format(player.Root.ChildWinRates()))
        print('Child Exploration Rates: {}'.format(player.Root.ChildPlays()))
        print(f'Time taken: {time.time()-start}')
        player.MoveRoot(state)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.dump_stats('profile.log')
    print(state)
    print(state.Winner())

if __name__ == "__main__":
    main()
    exit()
    now = time.time()
    for game in range(100):
        # print(f'Playing game {game+1}')
        dc = BoardState()
        # print(dc)
        while dc.Winner() == None:
            # choice = input("Enter a move: ")
            actions = dc.LegalActions()
            choice = np.random.choice(np.where(actions == 1)[0])
            try:
                # if ' ' in choice:
                #     choice = dc.move_to_int[choice]
                # print(choice)
                # if choice < 4032:
                #     print(dc.int_to_move[choice])
                # elif choice < 4208:
                #     print('PROMOTION')
                # else:
                #     print('CASTLE')
                # input()
                dc.ApplyAction(choice)
            except ValueError:
                print("The move you entered was illegal.")
            # print(dc)
        # print(dc.Winner())

    print(time.time()-now)
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