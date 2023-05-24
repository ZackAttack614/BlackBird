#include <iostream>
#include <array>
#include <algorithm>
#include <map>
#include <cmath>
#include <math.h>
#include <boost/math/constants/constants.hpp>
#include <string>
#include <cctype>
#include <random>
#include <chrono>
// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// namespace py = pybind11;

// typedef boost::multi_array<int, 2> array_type;
// typedef array_type::index index;
/*
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
*/

class GameState {
public:
    std::string EvalToString(std::string eval) {
        return eval;
    }
};

class BoardState : GameState {
public:
    const std::string GameType = "DragonChess";
    const static int NumLegalMoves = 802;
    std::map<int, int> piece_map;
    std::array<std::array<int, 8>, 8> board;
    std::map<int, std::array<int, 2>> piece_locs;
    int LastAction;
    int Player;
    int PreviousPlayer;
    int playedMoves;
    // std::array<int, 2> whiteKing;
    // std::array<int, 2> blackKing;
    bool _black_castle_kingside;
    bool _black_castle_queenside;

    //     int _is_legal_move(int loc_row, int loc_col, int new_row, int new_col) {
    //         int piece_type = abs(this->board[loc_row][loc_col]);
    //         if (piece_type == 5) {
    //             return _is_legal_move_king(loc_row, loc_col, new_row, new_col);
    //         }
    //         if (piece_type >= 41 || piece_type == 4) {
    //             return _is_legal_move_queen(loc_row, loc_col, new_row, new_col);
    //         }
    //         if (piece_type >= 33 || piece_type == 1 || piece_type == 8) {
    //             return _is_legal_move_rook(loc_row, loc_col, new_row, new_col);
    //         }
    //         if (piece_type >= 25 || piece_type == 3 || piece_type == 6) {
    //             return _is_legal_move_bishop(loc_row, loc_col, new_row, new_col);
    //         }
    //         if (piece_type >= 17 || piece_type == 2 || piece_type == 7) {
    //             return _is_legal_move_knight(loc_row, loc_col, new_row, new_col);
    //         }
    //         return _is_legal_move_pawn(loc_row, loc_col, new_row, new_col);
    //     };
    //     int _is_legal_move_pawn(int loc_row, int loc_col, int new_row, int new_col) {
    //         if (this->board[loc_row][loc_col] > 0) {
    //             if (loc_row == 6 && loc_col == new_col && new_row == 7 && this->board[7][new_col] == 0) {
    //                 return 2;
    //             }
    //             else if (loc_row == 6 && abs(loc_col - new_col) == 1 && new_row == 7 && this->board[7][new_col] < 0) {
    //                 return 2;
    //             }
    //             else if (loc_row == 1) {
    //                 if (loc_col == new_col) {
    //                     if (new_row == 3 && this->board[3][new_col] == 0 && this->board[2][new_col] == 0) {
    //                         return 1;
    //                     }
    //                     else if (new_row == 2 && this->board[2][new_col] == 0) {
    //                         return 1;
    //                     }
    //                     return 0;
    //                 }
    //             }
    //             else if (loc_row > 1 && loc_row < 6) {
    //                 if (loc_col == new_col && this->board[new_row][new_col] == 0 && new_row == loc_row + 1) {
    //                     return 1;
    //                 }
    //             }
    //             if (abs(loc_col - new_col) == 1) {
    //                 if (new_row == loc_row + 1) {
    //                     if (this->board[new_row][new_col] < 0) {
    //                         return 1;
    //                     }
    //                 }
    //             }
    //         }
    //         else {
    //             if (loc_row == 1 && loc_col == new_col && new_row == 0 && this->board[0][new_col] == 0) {
    //             return 2;
    // }
    //             else if (loc_row == 1 && abs(loc_col - new_col) == 1 && new_row == 0 && this->board[0][new_col] > 0) {
    //             return 2;
    // }
    //             else if (loc_row == 6) {
    //             if (loc_col == new_col) {
    //                 if (new_row == 4 && this->board[4][new_col] == 0 && this->board[5][new_col] == 0) {
    //                     return 1;
    //                 }
    //                 else if (new_row == 5 && this->board[5][new_col] == 0) {
    //                     return 1;
    //                 }
    //                 return 0;
    //             }
    // }
    //             else if (loc_row > 1 && loc_row < 6) {
    //             if (loc_col == new_col && this->board[new_row][new_col] == 0 && new_row == loc_row - 1) {
    //                 return 1;
    //             }
    // }
    //             if (abs(loc_col - new_col) == 1) {
    //                 if (new_row == loc_row - 1) {
    //                     if (this->board[new_row][new_col] > 0) {
    //                         return 1;
    //                     }
    //                 }
    //             }
    //         }
    //         return 0;
    //     };
    //     int _is_legal_move_rook(int loc_row, int loc_col, int new_row, int new_col) {
    //         if (loc_row != new_row && loc_col != new_col) {
    //             return 0;
    //         }
    //         if (new_row == loc_row) {
    //             for (int col = std::min(new_col, loc_col) + 1; col < std::max(new_col, loc_col); col++) {
    //                 if (this->board[new_row][col] != 0) {
    //                     return 0;
    //                 }
    //             }
    //         }
    //         else {
    //             for (int row = std::min(new_row, loc_row) + 1; row < std::max(new_row, loc_row); row++) {
    //                 if (this->board[row][new_col] != 0) {
    //                     return 0;
    //                 }
    //             }
    //         }
    //         return 1;
    //     };
    //     int _is_legal_move_bishop(int loc_row, int loc_col, int new_row, int new_col) {
    //         if (abs(loc_row - new_row) != abs(loc_col - new_col)) {
    //             return 0;
    //         }
    //         for (int diag = 1; diag < abs(loc_row - new_row); diag++) {
    //             if (this->board[loc_row + diag * (new_row > loc_row ? 1 : -1)][loc_col + diag * (new_col > loc_col ? 1 : -1)] != 0) {
    //                 return 0;
    //             }
    //         }
    //         return 1;
    //     };
    //     int _is_legal_move_queen(int loc_row, int loc_col, int new_row, int new_col) {
    //         return (this->_is_legal_move_bishop(loc_row, loc_col, new_row, new_col) + this->_is_legal_move_rook(loc_row, loc_col, new_row, new_col));
    //     };
    //     int _is_legal_move_king(int loc_row, int loc_col, int new_row, int new_col) {
    //         if (abs(loc_row - new_row) <= 1 && abs(loc_col - new_col) <= 1) {
    //             return 1;
    //         }
    //         return 0;
    //     };
    //     int _is_legal_move_knight(int loc_row, int loc_col, int new_row, int new_col) {
    //         if (abs(loc_row - new_row) == 1 && abs(loc_col - new_col) == 2) {
    //             return 1;
    //         }
    //         if (abs(loc_row - new_row) == 2 && abs(loc_col - new_col) == 1) {
    //             return 1;
    //         }
    //         return 0;
    //     };

    static int intsign(int val) {
        if (val == 0) {
            return 0;
        }
        return (val > 0) ? 1 : -1;
    }
    static int sign(double val) {
        return (val > 0) ? 1 : -1;
    }
    static std::vector<int> _legal_moves_any(std::array<std::array<int, 8>, 8> board, int loc_row, int loc_col, int color, bool black_castle_kingside, bool black_castle_queenside) {
        int piece = abs(board[loc_row][loc_col]);
        if (piece == 5) {
            std::vector<int> result = std::vector<int>();
            for (int dir = 0; dir < 8; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 45) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int new_row = loc_row + adjSin;
                int new_col = loc_col + adjCos;
                if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8 && intsign(board[new_row][new_col]) != color) {
                    result.push_back(184 + dir);
                }
            }
            if (color != 1) {
                if (loc_row == 7 && loc_col == 4 && black_castle_kingside && board[7][5] == 0 && board[7][6] == 0) {
                    result.push_back(800);
                }
                if (loc_row == 7 && loc_col == 4 && black_castle_queenside && board[7][1] == 0 && board[7][2] == 0 && board[7][3] == 0) {
                    result.push_back(801);
                }
            }
            return result;
        }
        if (piece == 1 || piece == 8) {
            std::vector<int> result = std::vector<int>();
            for (int dir = 0; dir < 4; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 90) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(28 * ((piece - 1) / 7) + 7 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(28 * ((piece - 1) / 7) + 7 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece == 2 || piece == 7) {
            std::vector<int> result = std::vector<int>();
            for (int drow : {-2, -1, 1, 2}) {
                int new_row = loc_row + drow;
                for (int dcol : {3 - abs(drow), -3 + abs(drow)}) {
                    int new_col = loc_col + dcol;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8 && intsign(board[new_row][new_col]) != color) {
                        result.push_back(56 + 8 * ((piece - 2) / 5) + intsign(drow) * (-dcol + 2 + (intsign(dcol) - 1) / 2) + (-7) * ((intsign(drow) - 1) / 2));
                    }
                }
            }
            return result;
        }
        if (piece == 3 || piece == 6) {
            std::vector<int> result = std::vector<int>();
            for (int dir = 0; dir < 4; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 90 + 45) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(72 + 28 * ((piece - 3) / 3) + 7 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(72 + 28 * ((piece - 3) / 3) + 7 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece == 4) {
            std::vector<int> result = std::vector<int>();
            for (int dir = 0; dir < 8; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 45) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(128 + 7 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(128 + 7 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece >= 41) {
            std::vector<int> result = std::vector<int>();
            int DEFact = 192 + 64 * (piece - 41);
            for (int dir = 0; dir < 8; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 45) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(DEFact + 8 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(DEFact + 8 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece >= 33) {
            std::vector<int> result = std::vector<int>();
            int DEFact = 192 + 64 * (piece - 33);
            for (int dir = 0; dir < 4; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 90) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(DEFact + 16 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(DEFact + 16 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece >= 25) {
            std::vector<int> result = std::vector<int>();
            int DEFact = 192 + 64 * (piece - 25);
            for (int dir = 0; dir < 4; dir++) {
                using namespace boost::math::double_constants;
                double theta = (dir * 90 + 45) * pi / 180;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                int count = 1;
                while (true) {
                    int new_row = loc_row + count * adjSin;
                    int new_col = loc_col + count * adjCos;
                    if (-1 < new_row && new_row < 8 && -1 < new_col && new_col < 8) {
                        if (intsign(board[new_row][new_col]) == 0) {
                            result.push_back(DEFact + 8 + 16 * dir + count - 1);
                            count++;
                            continue;
                        }
                        else if (intsign(board[new_row][new_col]) != color) {
                            result.push_back(DEFact + 8 + 16 * dir + count - 1);
                            break;
                        }
                        else {
                            break;
                        }
                    }
                    break;
                }
            }
            return result;
        }
        if (piece >= 17) {
            std::vector<int> result = std::vector<int>();
            int DEFact = 192 + 64 * (piece - 17);
            for (int drow : {-2, -1, 1, 2}) {
                int new_row = loc_row + drow;
                if (new_row < 0 || new_row > 7) {
                    continue;
                }
                for (int dcol : {3 - abs(drow), -3 + abs(drow)}) {
                    int new_col = loc_col + dcol;
                    if (-1 < new_col && new_col < 8 && intsign(board[new_row][new_col]) != color) {
                        result.push_back(DEFact + 7 + 8 * (intsign(drow) * (-dcol + 2 + (intsign(dcol) - 1) / 2) + (-7) * ((intsign(drow) - 1) / 2)));
                    }
                }
            }
            return result;
        }
        // otherwise, pawn
        std::vector<int> result = std::vector<int>();
        int DEFact = 192 + 64 * (piece - 9);
        if (color == 1) {
            if (loc_row != 6) {
                if (loc_col != 0) {
                    if (intsign(board[loc_row + 1][loc_col - 1]) == -1) {
                        result.push_back(DEFact + 24);
                    }
                }
                if (loc_col != 7) {
                    if (intsign(board[loc_row + 1][loc_col + 1]) == -1) {
                        result.push_back(DEFact + 8);
                    }
                }
                if (intsign(board[loc_row + 1][loc_col]) == 0) {
                    result.push_back(DEFact + 16);
                    if (loc_row == 1 && intsign(board[loc_row + 2][loc_col]) == 0) {
                        result.push_back(DEFact + 17);
                    }
                }
            }
            else {
                int PromDEFact = 704 + 12 * (piece - 9);
                if (loc_col != 0) {
                    if (intsign(board[loc_row + 1][loc_col - 1]) == -1) {
                        for (int promotion = 0; promotion < 4; promotion++) {
                            result.push_back(PromDEFact + promotion);
                        }
                    }
                }
                if (loc_col != 7) {
                    if (intsign(board[loc_row + 1][loc_col + 1]) == -1) {
                        for (int promotion = 0; promotion < 4; promotion++) {
                            result.push_back(PromDEFact + 8 + promotion);
                        }
                    }
                }
                if (intsign(board[loc_row + 1][loc_col]) == 0) {
                    for (int promotion = 0; promotion < 4; promotion++) {
                        result.push_back(PromDEFact + 4 + promotion);
                    }
                }
            }
        }
        else {
            if (loc_row != 1) {
                if (loc_col != 0) { // left capture
                    if (intsign(board[loc_row - 1][loc_col - 1]) == 1) {
                        result.push_back(DEFact + 40);
                    }
                }
                if (loc_col != 7) { // right capture
                    if (intsign(board[loc_row - 1][loc_col + 1]) == 1) {
                        result.push_back(DEFact + 56);
                    }
                }
                if (intsign(board[loc_row - 1][loc_col]) == 0) { // move up one
                    result.push_back(DEFact + 48);
                    if (loc_row == 6 && intsign(board[loc_row - 2][loc_col]) == 0) { // move up two
                        result.push_back(DEFact + 49);
                    }
                }
            }
            else {
                int PromDEFact = 704 + 12 * (piece - 9);
                if (loc_col != 0) { // left capture
                    if (intsign(board[loc_row - 1][loc_col - 1]) == 1) {
                        for (int promotion = 0; promotion < 4; promotion++) {
                            result.push_back(PromDEFact + promotion);
                        }
                    }
                }
                if (loc_col != 7) { // right capture
                    if (intsign(board[loc_row - 1][loc_col + 1]) == 1) {
                        for (int promotion = 0; promotion < 4; promotion++) {
                            result.push_back(PromDEFact + 8 + promotion);
                        }
                    }
                }
                if (intsign(board[loc_row - 1][loc_col]) == 0) { // move up one
                    for (int promotion = 0; promotion < 4; promotion++) {
                        result.push_back(PromDEFact + 4 + promotion);
                    }
                }
            }
        }
        return result;
    }
    //static py::array_t<int> py_legal_moves_any(py::array_t<int> board, int loc_row, int loc_col, int color, bool black_castle_kingside, bool black_castle_queenside) {
    //    std::array<std::array<int, 8>, 8> cBoard = board.cast<std::array<std::array<int, 8>, 8>>();
    //    std::vector<int> legalMoves = _legal_moves_any(cBoard, loc_row, loc_col, color, black_castle_kingside, black_castle_queenside);
    //    return py::cast(legalMoves);
    //}
    //static int squaresToMove(std::array<std::array<int, 8>, 8> board, int loc_row, int loc_col, int new_row, int new_col) {
    //    int piece = abs(board[loc_row][loc_col]);
    //};

    std::array<int, 802> LegalActions() {
        std::array<int, 802> legal = std::array<int, 802>();
        //std::array<int, 2> oppKing = this->piece_locs[(2*this->Player-3)*5];
        //int oppKingRow = oppKing[0];
        //int oppKingCol = oppKing[1];
        //for (int i = 0; i < 8; i++) {
        //    for (int j = 0; j < 8; j++) {
        //        if (this->_is_legal_move(i, j, oppKingRow, oppKingCol) > 0) {
        //            legal[squaresToMove(this->board, i, j, oppKingRow, oppKingCol)] = 1;
        //            return legal;
        //        }
        //    }
        //}
        for (int row_1 = 0; row_1 < 8; row_1++) {
            for (int col_1 = 0; col_1 < 8; col_1++) {
                if (intsign(this->board[row_1][col_1]) == -2 * this->Player + 3) {
                    for (int move : _legal_moves_any(this->board, row_1, col_1, -2 * this->Player + 3, this->_black_castle_kingside, this->_black_castle_queenside)) {
                        legal[move] = 1;
                    }
                }
            }
        }
        return legal;
    };
    // py::array_t<int> py_LegalActions() {
    //     std::array<int, 802> arr = LegalActions();
    //     return py::array_t<int>(arr.size(), arr.data());
    // };
    int NumLegalActions() {
        // std::array<int, 2> oppKing = this->piece_locs[(2 * this->Player - 3) * 5];
        // int oppKingRow = oppKing[0];
        // int oppKingCol = oppKing[1];
        // for (int i = 0; i < 8; i++) {
        //     for (int j = 0; j < 8; j++) {
        //         if (this->_is_legal_move(i, j, oppKingRow, oppKingCol) > 0) {
        //             return 1;
        //         }
        //     }
        // }
        int numActions = 0;
        for (int row_1 = 0; row_1 < 8; row_1++) {
            for (int col_1 = 0; col_1 < 8; col_1++) {
                if (intsign(this->board[row_1][col_1]) == -2 * this->Player + 3) {
                    numActions += _legal_moves_any(this->board, row_1, col_1, -2 * this->Player + 3, this->_black_castle_kingside, this->_black_castle_queenside).size();
                }
            }
        }
        return numActions;
    };
    void ApplyAction(int action) {
        if (action < 28) {
            std::array<int, 2> piece = this->piece_locs[-2 * this->Player + 3];
            using namespace boost::math::double_constants;
            double theta = (action / 7) * pi / 2;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + (action % 7 + 1) * adjSin, piece[1] + (action % 7 + 1) * adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 56) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 8];
            using namespace boost::math::double_constants;
            double theta = ((action - 28) / 7) * pi / 2;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + (action % 7 + 1) * adjSin, piece[1] + (action % 7 + 1) * adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 64) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 2];
            std::map<int, int> ydists = { {0, 2}, {1, 1}, {2, -1}, {3, -2}, {4, -2}, {5, -1}, {6, 1}, {7, 2} };
            std::map<int, int> xdists = { {0, 1}, {1, 2}, {2, 2}, {3, 1}, {4, -1}, {5, -2}, {6, -2}, {7, -1} };
            std::array<int, 2> newLoc = { piece[0] + xdists[action - 56], piece[1] + ydists[action - 56] };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 72) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 7];
            std::map<int, int> ydists = { {0, 2}, {1, 1}, {2, -1}, {3, -2}, {4, -2}, {5, -1}, {6, 1}, {7, 2} };
            std::map<int, int> xdists = { {0, 1}, {1, 2}, {2, 2}, {3, 1}, {4, -1}, {5, -2}, {6, -2}, {7, -1} };
            std::array<int, 2> newLoc = { piece[0] + xdists[action - 64], piece[1] + ydists[action - 64] };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 100) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 3];
            using namespace boost::math::double_constants;
            double theta = (90 * ((action - 72) / 7) + 45) * pi / 180;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + ((action - 72) % 7 + 1) * adjSin, piece[1] + ((action - 72) % 7 + 1) * adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 128) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 6];
            using namespace boost::math::double_constants;
            double theta = (90 * ((action - 100) / 7) + 45) * pi / 180;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + ((action - 100) % 7 + 1) * adjSin, piece[1] + ((action - 100) % 7 + 1) * adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 184) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 4];
            using namespace boost::math::double_constants;
            double theta = ((action - 128) / 7) * pi / 4;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + ((action - 128) % 7 + 1) * adjSin, piece[1] + ((action - 128) % 7 + 1) * adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 192) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * 5];
            using namespace boost::math::double_constants;
            double theta = ((action - 184) % 8) * pi / 4;
            double c = cos(theta);
            double s = sin(theta);
            int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
            int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
            std::array<int, 2> newLoc = { piece[0] + adjSin, piece[1] + adjCos };
            this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
        }
        else if (action < 704) {
            std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * ((action - 192) / 64 + 9)];
            int pieceNumber = (-2 * this->Player + 3) * ((action - 192) / 64 + 9);
            int count = 0;
            while (piece[0] == -1) {
                piece = this->piece_locs[(-2 * this->Player + 3) * ((action - 192) / 64 + 17 + 8 * count)];
                count++;
                pieceNumber += (-2 * this->Player + 3) * 8;
            }
            if (action % 8 == 7) {
                std::map<int, int> ydists = { {0, 2}, {1, 1}, {2, -1}, {3, -2}, {4, -2}, {5, -1}, {6, 1}, {7, 2} };
                std::map<int, int> xdists = { {0, 1}, {1, 2}, {2, 2}, {3, 1}, {4, -1}, {5, -2}, {6, -2}, {7, -1} };
                std::array<int, 2> newLoc = { piece[0] + xdists[(action % 64) / 8], piece[1] + ydists[(action % 64) / 8] };
                this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
            }
            else {
                using namespace boost::math::double_constants;
                double theta = ((action % 64) / 8) * pi / 4;
                double c = cos(theta);
                double s = sin(theta);
                int adjCos = (abs(c) < 1e-6) ? 0 : sign(c);
                int adjSin = (abs(s) < 1e-6) ? 0 : sign(s);
                std::array<int, 2> newLoc = { piece[0] + (action % 8 + 1) * adjSin, piece[1] + (action % 8 + 1) * adjCos };
                this->Move(piece[0], piece[1], newLoc[0], newLoc[1], 0, false);
            }
        }
        else {
            if (action >= 800) { // only black can castle
                action -= 800;
                std::array<int, 2> newLoc = { 7, -4 * (action % 2) + 6 };
                this->Move(7, 4, 7, newLoc[1], 0, true);
                action += 800;
            }
            else {
                action -= 704;
                int promotePiece = 17 + (action / 12) + 8 * (action % 4);
                std::array<int, 2> piece = this->piece_locs[(-2 * this->Player + 3) * (action / 12 + 9)];
                if (action % 12 < 4) {
                    std::array<int, 2> newLoc = { piece[0] + (-2 * this->Player + 3), piece[1] - 1 };
                    this->Move(piece[0], piece[1], newLoc[0], newLoc[1], (-2 * this->Player + 3) * promotePiece, false);
                }
                else if (action % 12 < 8) {
                    std::array<int, 2> newLoc = { piece[0] + (-2 * this->Player + 3), piece[1] };
                    this->Move(piece[0], piece[1], newLoc[0], newLoc[1], (-2 * this->Player + 3) * promotePiece, false);
                }
                else {
                    std::array<int, 2> newLoc = { piece[0] + (-2 * this->Player + 3), piece[1] + 1 };
                    this->Move(piece[0], piece[1], newLoc[0], newLoc[1], (-2 * this->Player + 3) * promotePiece, false);
                }
                action += 704;
            }
        }

        this->playedMoves++;
        this->LastAction = action;
    };
    void Move(int loc_row, int loc_col, int new_row, int new_col, int promote, bool castle) {
        if (this->board[new_row][new_col] != 0) {
            this->piece_locs[this->board[new_row][new_col]] = { -1, -1 };
        }
        if (promote != 0) {
            this->board[new_row][new_col] = promote;
            this->piece_locs[promote] = { new_row, new_col };
            this->piece_locs[this->board[loc_row][loc_col]] = { -1, -1 };
        }
        else if (castle) {
            this->board[new_row][new_col] = this->board[loc_row][loc_col];
            this->board[new_row][new_col / 2 + 2] = this->board[new_row][new_col * 7 / 4 - 3];
            this->piece_locs[this->board[new_row][new_col / 2 + 2]] = { new_row, new_col / 2 + 2 };
            this->board[new_row][new_col * 7 / 4 - 3] = 0;
            this->piece_locs[this->board[loc_row][loc_col]] = { new_row, new_col };
        }
        else {
            this->board[new_row][new_col] = this->board[loc_row][loc_col];
            this->piece_locs[this->board[loc_row][loc_col]] = { new_row, new_col };
        }
        this->board[loc_row][loc_col] = 0;
        if (this->PreviousPlayer == 1 && this->Player == 1) {
            this->Player = 2;
        }
        else {
            this->PreviousPlayer = this->Player;
            this->Player = 1;
        }
        if (this->board[7][4] != -5) {
            this->_black_castle_kingside = false;
            this->_black_castle_queenside = false;
        }
        else if (this->board[7][7] != -8) {
            this->_black_castle_kingside = false;
        }
        else if (this->board[7][0] != -1) {
            this->_black_castle_queenside = false;
        }
    };
    std::string to_string() {
        std::string repr = "-----------------\n";
        for (int row = 0; row < 8; row++) {
            std::string row_repr = "|";
            for (int col = 0; col < 8; col++) {
                int piece = abs(board[7 - row][col]);
                char nextChar = ' ';
                if (piece != 0) {
                    if (piece == 1 || piece == 8 || (33 <= piece && piece <= 40)) {
                        nextChar = 'r';
                    }
                    else if (piece == 2 || piece == 7 || (17 <= piece && piece <= 24)) {
                        nextChar = 'n';
                    }
                    else if (piece == 3 || piece == 6 || (25 <= piece && piece <= 32)) {
                        nextChar = 'b';
                    }
                    else if (piece == 4 || (41 <= piece && piece <= 48)) {
                        nextChar = 'q';
                    }
                    else if (piece == 5) {
                        nextChar = 'k';
                    }
                    else {
                        nextChar = 'p';
                    }
                }
                if (board[7 - row][col] > 0) {
                    nextChar = toupper(nextChar);
                }
                row_repr += nextChar;
                row_repr += '|';
            }
            repr += row_repr + "\n";
        }
        return repr + "-----------------";
    };
    BoardState() : board(std::array<std::array<int, 8>, 8>()), Player(1), PreviousPlayer(0), _black_castle_kingside(true), _black_castle_queenside(true), playedMoves(0), LastAction(-1) {
        piece_map[5] = 10;
        piece_map[-5] = 11;
        piece_map[2] = 4;
        piece_map[7] = 4;
        piece_map[-2] = 5;
        piece_map[-7] = 5;
        piece_map[3] = 6;
        piece_map[6] = 6;
        piece_map[-3] = 7;
        piece_map[-6] = 7;
        piece_map[1] = 2;
        piece_map[8] = 2;
        piece_map[-1] = 3;
        piece_map[-8] = 3;
        piece_map[4] = 8;
        piece_map[-4] = 9;
        for (int i = 9; i < 17; i++) {
            piece_map[i] = 0;
            piece_map[i - 25] = 1;
        }
        for (int i = 17; i < 25; i++) {
            piece_map[i] = 4;
            piece_map[i - 41] = 5;
        }
        for (int i = 25; i < 33; i++) {
            piece_map[i] = 6;
            piece_map[i - 57] = 7;
        }
        for (int i = 33; i < 41; i++) {
            piece_map[i] = 2;
            piece_map[i - 73] = 3;
        }
        for (int i = 41; i < 49; i++) {
            piece_map[i] = 8;
            piece_map[i - 89] = 9;
        }
        board[0][4] = 5;
        piece_locs[5] = { 0, 4 };
        board[1][3] = 12;
        board[1][4] = 13;
        board[1][5] = 14;
        for (int i = 3; i < 6; i++) {
            piece_locs[9 + i] = { 1, i };
        }
        for (int missing : {1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16}) {
            piece_locs[missing] = { -1, -1 };
        }
        for (int i = 0; i < 8; i++) {
            board[6][i] = -i - 9;
            piece_locs[-i - 9] = { 6, i };
            board[7][i] = -i - 1;
            piece_locs[-i - 1] = { 7, i };
        }
        for (int i = 17; i < 49; i++) {
            piece_locs[i] = { -1, -1 };
            piece_locs[-i] = { -1, -1 };
        }
    };

    std::array<std::array<int, 8>, 8> Board() {
        return board;
    };
    // py::array_t<int> py_LegalActionShape() {
    //     std::array<int, 802> arr = std::array<int, 802>();
    //     return py::array_t<int>(arr.size(), arr.data());
    // };
    BoardState Copy() {
        BoardState copy;
        copy.Player = this->Player;
        copy.LastAction = this->LastAction;
        copy.PreviousPlayer = this->PreviousPlayer;
        copy._black_castle_kingside = this->_black_castle_kingside;
        copy._black_castle_queenside = this->_black_castle_queenside;
        copy.board = this->board;
        copy.playedMoves = this->playedMoves;
        copy.piece_locs = this->piece_locs;
        return copy;
    };
    bool eq(BoardState b) {
        if (b.board == this->board && b.Player == this->Player && b.PreviousPlayer == this->PreviousPlayer && b._black_castle_kingside == this->_black_castle_kingside && b._black_castle_queenside == this->_black_castle_queenside) {
            return true;
        }
        return false;
    };

    int Winner() {
        if (this->piece_locs[-5][0] == -1) {
            return 1;
        }
        if (this->piece_locs[5][0] == -1) {
            return 2;
        }
        if (this->playedMoves >= 80) {
            return 0;
        }
        return -1;
    }
};

int main() {
    BoardState b;
    b.ApplyAction(401);
    BoardState s;
    s.ApplyAction(400);
    s.ApplyAction(400);
    std::cout << (b.eq(s));
    return 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        BoardState b;
        //std::cout << b.to_string() << std::endl;
        while (b.Winner() == -1) {
            std::array<int, 802> v = b.LegalActions();
            std::vector<int> legalMoves = std::vector<int>();
            for (int i = 0; i < 802; i++) {
                if (v[i] == 1) {
                    legalMoves.push_back(i);
                }
            }
            std::random_device rd;
            std::mt19937 rng(rd());
            std::uniform_int_distribution<int> dist(0, legalMoves.size() - 1);

            int index = dist(rng);
            b.ApplyAction(legalMoves[index]);
            //std::cout << "Next move: " << legalMoves[index] << " chosen out of " << legalMoves.size() << " legal moves." << std::endl;
            //std::cout << b.to_string() << std::endl;
            //std::string x;
            //std::getline(std::cin, x);
        }
        std::cout << "Winner: " << b.Winner() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Duration: " << duration / 1000000.0 << " seconds" << std::endl;
    //b.ApplyAction(401);
    //b.ApplyAction(464);
    //b.ApplyAction(497);
    //b.ApplyAction(392);
    //std::cout << b.to_string() << std::endl;
    ////for (const auto& kvp : b.piece_locs) {
    ////    std::cout << "Key: " << kvp.first << ", Value: " << kvp.second[0] << ", " << kvp.second[1] << std::endl;
    ////}
    //std::array<int, 802> v = b.LegalActions();
    //for (int i = 0; i < 802; i++) {
    //    if (v[i] == 1) {
    //        std::cout << i << " ";
    //    }
    //}
    //return 0;
}