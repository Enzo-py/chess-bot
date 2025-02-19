from models.engine import Engine
import chess
import time
from collections import namedtuple
from itertools import count

class SunfishAI(Engine):
    """
    Implementation of Sunfish chess algorithm within the Engine framework.
    """

    __author__ = "Shashuat ..."
    __description__ = "Chess AI that uses the Sunfish engine."
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Piece values from Sunfish
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 280,
            chess.BISHOP: 320,
            chess.ROOK: 479,
            chess.QUEEN: 929,
            chess.KING: 60000
        }
        
        # Initialize piece-square tables (simplified for compatibility)
        self.pst = self._initialize_pst()
        
        # Constants for mate detection
        self.MATE_LOWER = self.piece_values[chess.KING] - 10 * self.piece_values[chess.QUEEN]
        self.MATE_UPPER = self.piece_values[chess.KING] + 10 * self.piece_values[chess.QUEEN]
        
    def _initialize_pst(self):
        """Initialize piece-square tables from Sunfish values."""
        pst = {}
        
        # Pawn table
        pst[chess.PAWN] = [
            0,   0,   0,   0,   0,   0,   0,   0,
            78,  83,  86,  73,  102, 82,  85,  90,
            7,   29,  21,  44,  40,  31,  44,  7,
            -17, 16,  -2,  15,  14,  0,   15,  -13,
            -26, 3,   10,  9,   6,   1,   0,   -23,
            -22, 9,   5,   -11, -10, -2,  3,   -19,
            -31, 8,   -7,  -37, -36, -14, 3,   -31,
            0,   0,   0,   0,   0,   0,   0,   0
        ]
        
        # Knight table
        pst[chess.KNIGHT] = [
            -66, -53, -75, -75, -10, -55, -58, -70,
            -3,  -6,  100, -36, 4,   62,  -4,  -14,
            10,  67,  1,   74,  73,  27,  62,  -2,
            24,  24,  45,  37,  33,  41,  25,  17,
            -1,  5,   31,  21,  22,  35,  2,   0,
            -18, 10,  13,  22,  18,  15,  11,  -14,
            -23, -15, 2,   0,   2,   0,   -23, -20,
            -74, -23, -26, -24, -19, -35, -22, -69
        ]
        
        # Bishop table
        pst[chess.BISHOP] = [
            -59, -78, -82, -76, -23, -107, -37, -50,
            -11, 20,  35,  -42, -39, 31,   2,   -22,
            -9,  39,  -32, 41,  52,  -10,  28,  -14,
            25,  17,  20,  34,  26,  25,   15,  10,
            13,  10,  17,  23,  17,  16,   0,   7,
            14,  25,  24,  15,  8,   25,   20,  15,
            19,  20,  11,  6,   7,   6,    20,  16,
            -7,  2,   -15, -12, -14, -15,  -10, -10
        ]
        
        # Rook table
        pst[chess.ROOK] = [
            35,  29,  33,  4,   37,  33,  56,  50,
            55,  29,  56,  67,  55,  62,  34,  60,
            19,  35,  28,  33,  45,  27,  25,  15,
            0,   5,   16,  13,  18,  -4,  -9,  -6,
            -28, -35, -16, -21, -13, -29, -46, -30,
            -42, -28, -42, -25, -25, -35, -26, -46,
            -53, -38, -31, -26, -29, -43, -44, -53,
            -30, -24, -18, 5,   -2,  -18, -31, -32
        ]
        
        # Queen table
        pst[chess.QUEEN] = [
            6,   1,   -8,  -104, 69,  24,  88,  26,
            14,  32,  60,  -10,  20,  76,  57,  24,
            -2,  43,  32,  60,   72,  63,  43,  2,
            1,   -16, 22,  17,   25,  20,  -13, -6,
            -14, -15, -2,  -5,   -1,  -10, -20, -22,
            -30, -6,  -13, -11,  -16, -11, -16, -27,
            -36, -18, 0,   -19,  -15, -15, -21, -38,
            -39, -30, -31, -13,  -31, -36, -34, -42
        ]
        
        # King table
        pst[chess.KING] = [
            4,   54,  47,  -99, -99, 60,  83,  -62,
            -32, 10,  55,  56,  56,  55,  10,  3,
            -62, 12,  -57, 44,  -67, 28,  37,  -31,
            -55, 50,  11,  -4,  -19, 13,  0,   -49,
            -55, -43, -52, -28, -51, -47, -8,  -50,
            -47, -42, -43, -79, -64, -32, -29, -32,
            -4,  3,   -14, -50, -57, -18, 13,  4,
            17,  30,  -3,  -14, 6,   -1,  40,  18
        ]
        
        return pst

    def play(self) -> chess.Move:
        """
        Select the best move using Sunfish's alpha-beta search.
        """
        start_time = time.time()
        think_time = 1.0  # Maximum thinking time in seconds
        
        best_move = None
        depth = 1
        
        # Iterative deepening
        while time.time() - start_time < think_time:
            try:
                score, move = self._alpha_beta(self.game.board, depth, -self.MATE_UPPER, self.MATE_UPPER)
                if move is not None:
                    best_move = move
                depth += 1
            except TimeoutError:
                break
        
        if best_move is None:
            # Fallback to any legal move
            return list(self.game.board.legal_moves)[0]
            
        return best_move

    def _alpha_beta(self, board, depth, alpha, beta):
        """
        Alpha-beta search implementation adapted from Sunfish.
        """
        if depth == 0:
            return self._evaluate_position(board), None
            
        best_move = None
        moves = list(board.legal_moves)
        
        if not moves:
            if board.is_checkmate():
                return -self.MATE_UPPER, None
            return 0, None  # Stalemate
            
        # Move ordering - captures first
        moves.sort(key=lambda m: (
            1000 if board.is_capture(m) else 0 +
            500 if board.gives_check(m) else 0 +
            self._get_move_value(board, m)
        ), reverse=True)
        
        for move in moves:
            board.push(move)
            score, _ = self._alpha_beta(board, depth - 1, -beta, -alpha)
            score = -score
            board.pop()
            
            if score > alpha:
                alpha = score
                best_move = move
                
            if alpha >= beta:
                break
                
        return alpha, best_move

    def _evaluate_position(self, board) -> float:
        """
        Evaluate the current position using material and piece-square tables.
        """
        if board.is_checkmate():
            return -self.MATE_UPPER
            
        if board.is_stalemate():
            return 0
            
        score = 0
        
        # Material and piece-square table evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            value = self.piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
                score += self.pst[piece.piece_type][square]
            else:
                score -= value
                score -= self.pst[piece.piece_type][chess.square_mirror(square)]
                
        # Mobility evaluation
        score += len(list(board.legal_moves)) * 10
        
        # King safety
        w_king_square = board.king(chess.WHITE)
        b_king_square = board.king(chess.BLACK)
        
        if w_king_square:
            score += len(list(board.attackers(chess.WHITE, w_king_square))) * -50
        if b_king_square:
            score -= len(list(board.attackers(chess.BLACK, b_king_square))) * -50
            
        return score if board.turn == chess.WHITE else -score

    def _get_move_value(self, board, move):
        """
        Get the preliminary value of a move for move ordering.
        """
        score = 0
        
        # Capturing moves
        if board.is_capture(move):
            victim_piece = board.piece_at(move.to_square)
            attacker_piece = board.piece_at(move.from_square)
            if victim_piece and attacker_piece:
                score = 10 * self.piece_values[victim_piece.piece_type] - self.piece_values[attacker_piece.piece_type]
                
        # Promotion moves
        if move.promotion:
            score += self.piece_values[move.promotion]
            
        # Center control
        if move.to_square in [27, 28, 35, 36]:  # e4, d4, e5, d5
            score += 30
            
        # Castle moves
        if board.is_castling(move):
            score += 60
            
        return score