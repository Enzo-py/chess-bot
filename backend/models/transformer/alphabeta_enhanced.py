import chess
import time
import numpy as np
import random
from models.deep_engine import DeepEngine
from models.greedy.greedy_ai import GreedyAI
from models.greedy.greedy_exploration import GreedyExplorationAI
from collections import OrderedDict
from models.transformer.transformer import GenerativeHead, Decoder, ChessTransformerEncoder, BoardEvaluator


class TransformerAlphaBetaEnhanced(DeepEngine):
    """
    Enhanced Stockfish-inspired chess AI with advanced search techniques.
    
    Features:
    - Alpha-beta pruning with iterative deepening
    - Transposition table
    - Move ordering
    - Quiescence search
    - Late Move Reduction (LMR)
    - Null Move Pruning
    - Enhanced evaluation function
    """
    
    __author__ = "Shashwat SHARMA"
    __description__ = "Stockfish-inspired chess AI again which has alpha-beta pruning"
    __weights__ = 'TransformerScore'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.time_limit = 0.9  # Maximum time in seconds (we want to play chess fast)
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330, 
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        # Initialize piece-square tables 
        self.initialize_tables()
        
        # Initialize fallback engines
        self.greedy_exploration = None
        self.greedy = None
        
        # For time management
        self.start_time = 0
        self.nodes_evaluated = 0
        self.should_stop = False
        
        # Transposition table (position hash -> [depth, value, flag, best_move])
        # flag: 0 = exact, 1 = lower bound, 2 = upper bound
        self.tt_size = 1000000  # Size of transposition table
        self.transposition_table = OrderedDict()
        
        # For move ordering
        self.killer_moves = [[None, None] for _ in range(100)]  # Store 2 killer moves per ply (up to 100 ply)
        self.history_table = {}  # Store history heuristic scores
        
        # For null move pruning
        self.R = 2  # Null move reduction factor
        
        # For late move reduction
        self.lmr_full_depth_moves = 4  # Number of moves to search at full depth
        self.lmr_reduction_limit = 3    # Don't reduce below this depth
        
        # Opening book
        self.opening_book = self.initialize_opening_book()

        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessTransformerEncoder())
        self.set(head_name="decoder", head=Decoder())
        self.confidence = 0.5

    def evaluate(self, game):
        self.setup()
        self.game = game
        self.color = game.board.turn
        prob = self.evaluate_position(game.board)
        if self.color == chess.BLACK:
            return prob, 1 - prob
        return 1 - prob, prob

        
    def initialize_opening_book(self):
        """Initialize a simple opening book with common opening moves"""
        book = {}
        
        # Starting position
        start_pos = chess.Board().fen()
        book[start_pos] = ["e2e4", "d2d4", "c2c4", "g1f3"]  # Common first moves
        
        # After 1.e4
        e4_pos = chess.Board()
        e4_pos.push_san("e4")
        book[e4_pos.fen()] = ["e7e5", "c7c5", "e7e6", "c7c6"]  # Common responses to e4
        
        # After 1.d4
        d4_pos = chess.Board()
        d4_pos.push_san("d4")
        book[d4_pos.fen()] = ["d7d5", "g8f6", "e7e6", "c7c5"]  # Common responses to d4
        
        return book
        
    def initialize_tables(self):
        """Initialize piece-square tables for positional evaluation"""
        # Pawn table encourages center control and advancement
        self.pawn_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ])
        
        # Knight table encourages knights to stay near the center
        self.knight_table = np.array([
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ])
        
        # Bishop table encourages bishops to control diagonals
        self.bishop_table = np.array([
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -10,  0,  5,  0,  0,  5,  0,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ])
        
        # Rook table encourages rooks to control open files
        self.rook_table = np.array([
            0,  0,  0,  0,  0,  0,  0,  0,
            5, 10, 10, 10, 10, 10, 10,  5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            -5,  0,  0,  0,  0,  0,  0, -5,
            0,  0,  0,  5,  5,  0,  0,  0
        ])
        
        # Queen table encourages the queen to stay near the center but not too early
        self.queen_table = np.array([
            -20,-10,-10, -5, -5,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5,  5,  5,  5,  0,-10,
            -5,  0,  5,  5,  5,  5,  0, -5,
            0,  0,  5,  5,  5,  5,  0, -5,
            -10,  5,  5,  5,  5,  5,  0,-10,
            -10,  0,  5,  0,  0,  0,  0,-10,
            -20,-10,-10, -5, -5,-10,-10,-20
        ])
        
        # King table for middlegame - encourages king safety
        self.king_middle_table = np.array([
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -30,-40,-40,-50,-50,-40,-40,-30,
            -20,-30,-30,-40,-40,-30,-30,-20,
            -10,-20,-20,-20,-20,-20,-20,-10,
            20, 20,  0,  0,  0,  0, 20, 20,
            20, 30, 10,  0,  0, 10, 30, 20
        ])
        
        # King table for endgame - encourages king activity
        self.king_end_table = np.array([
            -50,-40,-30,-20,-20,-30,-40,-50,
            -30,-20,-10,  0,  0,-10,-20,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 30, 40, 40, 30,-10,-30,
            -30,-10, 20, 30, 30, 20,-10,-30,
            -30,-30,  0,  0,  0,  0,-30,-30,
            -50,-30,-30,-30,-30,-30,-30,-50
        ])
        
        # Flip tables for black pieces (mirrored across ranks)
        self.tables = {
            chess.PAWN: {chess.WHITE: self.pawn_table, chess.BLACK: np.flip(self.pawn_table, 0)},
            chess.KNIGHT: {chess.WHITE: self.knight_table, chess.BLACK: np.flip(self.knight_table, 0)},
            chess.BISHOP: {chess.WHITE: self.bishop_table, chess.BLACK: np.flip(self.bishop_table, 0)},
            chess.ROOK: {chess.WHITE: self.rook_table, chess.BLACK: np.flip(self.rook_table, 0)},
            chess.QUEEN: {chess.WHITE: self.queen_table, chess.BLACK: np.flip(self.queen_table, 0)},
            chess.KING: {
                chess.WHITE: {
                    'middle': self.king_middle_table, 
                    'end': self.king_end_table
                }, 
                chess.BLACK: {
                    'middle': np.flip(self.king_middle_table, 0), 
                    'end': np.flip(self.king_end_table, 0)
                }
            }
        }
    
    def is_endgame(self, board):
        """Determine if the position is in the endgame phase"""
        # Check material count - endgame is when both sides have ≤ a queen or less in material
        # (excluding kings and pawns)
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type in [chess.KING, chess.PAWN]:
                continue
            
            if piece.color == chess.WHITE:
                white_material += self.piece_values[piece.piece_type]
            else:
                black_material += self.piece_values[piece.piece_type]
        
        return white_material <= 900 and black_material <= 900
    
    def evaluate_position(self, board):
        """
        Evaluate the board position.
        Positive scores favor white, negative favor black.
        """
        if board.is_checkmate():
            # Maximum value, accounting for the side to move
            return -20000 if board.turn else 20000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Draw
        
        score = 0
        is_endgame = self.is_endgame(board)
        
        # Material and piece-square table evaluation
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
                
            piece_type = piece.piece_type
            color = piece.color
            
            # Material count
            material_value = self.piece_values[piece_type]
            score += material_value if color == chess.WHITE else -material_value
            
            # Piece-square tables (positional evaluation)
            if piece_type == chess.KING:
                # Use different tables for king depending on game phase
                table_type = 'end' if is_endgame else 'middle'
                position_value = self.tables[piece_type][color][table_type][square]
            else:
                position_value = self.tables[piece_type][color][square]
            
            score += position_value if color == chess.WHITE else -position_value
        
        # Mobility (count of legal moves)
        current_turn = board.turn
        
        # Save current turn
        board.turn = chess.WHITE
        white_mobility = len(list(board.legal_moves))
        
        # Switch to black's perspective
        board.turn = chess.BLACK
        black_mobility = len(list(board.legal_moves))
        
        # Restore original turn
        board.turn = current_turn
        
        # Add mobility score (5 points per legal move)
        score += 5 * (white_mobility - black_mobility)
        
        # Pawn structure evaluation
        score += self.evaluate_pawn_structure(board)
        
        # King safety
        score += self.evaluate_king_safety(board)
        
        # Advanced evaluation: control of center
        score += self.evaluate_center_control(board)
        
        # Bishop pair bonus
        score += self.evaluate_bishop_pair(board)
        
        # Rook on open file bonus
        score += self.evaluate_rooks_on_open_files(board)
        
        # Knight outposts
        score += self.evaluate_knight_outposts(board)
        
        # Return score normalized to centipawns (1 pawn = 100 centipawns)
        
        ai_score = self.predict(head='board_evaluation')[int(self.color)]
        # normalise score using sigmoid function
        score = 1 / (1 + np.exp(-score))
        score = score * self.confidence + ai_score * (1 - self.confidence)
        return score
    
    def evaluate_pawn_structure(self, board):
        """Evaluate pawn structure: doubled/isolated/passed pawns"""
        score = 0
        
        # Check for doubled pawns (penalty)
        for file_idx in range(8):
            white_pawns_on_file = 0
            black_pawns_on_file = 0
            
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns_on_file += 1
                    else:
                        black_pawns_on_file += 1
            
            # Penalty for doubled pawns (-20 per doubled pawn)
            if white_pawns_on_file > 1:
                score -= (white_pawns_on_file - 1) * 20
            if black_pawns_on_file > 1:
                score += (black_pawns_on_file - 1) * 20
        
        # Check for isolated pawns
        for color in [chess.WHITE, chess.BLACK]:
            for square in board.pieces(chess.PAWN, color):
                file_idx = chess.square_file(square)
                
                # Check if there are friendly pawns on adjacent files
                has_friendly_pawn_adjacent = False
                
                # Check left file
                if file_idx > 0:
                    for rank in range(8):
                        adj_square = chess.square(file_idx - 1, rank)
                        piece = board.piece_at(adj_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            has_friendly_pawn_adjacent = True
                            break
                
                # Check right file
                if file_idx < 7 and not has_friendly_pawn_adjacent:
                    for rank in range(8):
                        adj_square = chess.square(file_idx + 1, rank)
                        piece = board.piece_at(adj_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == color:
                            has_friendly_pawn_adjacent = True
                            break
                
                if not has_friendly_pawn_adjacent:
                    # Isolated pawn penalty (-15)
                    score += -15 if color == chess.WHITE else 15
        
        # Check for passed pawns
        for color in [chess.WHITE, chess.BLACK]:
            opponent_color = not color
            for square in board.pieces(chess.PAWN, color):
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)
                
                # Direction of pawn movement
                direction = 1 if color == chess.WHITE else -1
                
                # Check if this is a passed pawn
                is_passed = True
                
                # Check if there are any opponent pawns that can block or capture this pawn
                for check_file in [file_idx - 1, file_idx, file_idx + 1]:
                    if check_file < 0 or check_file > 7:
                        continue
                        
                    # Check all ranks in front of the pawn
                    for check_rank in range(rank_idx + direction, 8 if direction > 0 else -1, direction):
                        if check_rank < 0 or check_rank > 7:
                            continue
                            
                        check_square = chess.square(check_file, check_rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN and piece.color == opponent_color:
                            is_passed = False
                            break
                    
                    if not is_passed:
                        break
                
                if is_passed:
                    # Bonus for passed pawn (value increases as pawn advances)
                    rank_bonus = rank_idx if color == chess.WHITE else 7 - rank_idx
                    passed_value = 20 + 10 * rank_bonus
                    score += passed_value if color == chess.WHITE else -passed_value
        
        return score
    
    def evaluate_king_safety(self, board):
        """Evaluate king safety based on pawn shield and attacker proximity"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            # Get king position
            king_square = board.king(color)
            if king_square is None:  # Shouldn't happen in a legal position
                continue
                
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            # Evaluate pawn shield (pawns near the king)
            pawn_shield_count = 0
            
            # Define area in front of king to check for pawn shield
            shield_offsets = []
            if color == chess.WHITE:
                if king_rank == 0:  # King on first rank
                    shield_offsets = [(0, 1), (-1, 1), (1, 1)]
                else:
                    shield_offsets = [(0, 1), (-1, 1), (1, 1), (0, 0), (-1, 0), (1, 0)]
            else:  # BLACK
                if king_rank == 7:  # King on last rank
                    shield_offsets = [(0, -1), (-1, -1), (1, -1)]
                else:
                    shield_offsets = [(0, -1), (-1, -1), (1, -1), (0, 0), (-1, 0), (1, 0)]
            
            # Count pawns in shield positions
            for file_offset, rank_offset in shield_offsets:
                check_file = king_file + file_offset
                check_rank = king_rank + rank_offset
                
                # Skip if out of board
                if check_file < 0 or check_file > 7 or check_rank < 0 or check_rank > 7:
                    continue
                
                check_square = chess.square(check_file, check_rank)
                piece = board.piece_at(check_square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    pawn_shield_count += 1
            
            # Bonus for each pawn in the shield (10 points per pawn)
            shield_value = pawn_shield_count * 10
            score += shield_value if color == chess.WHITE else -shield_value
            
            # Penalty for open files toward the king
            if king_file > 0 and king_file < 7:  # Not on a or h file
                # Check if king's file and adjacent files are open
                for check_file in [king_file - 1, king_file, king_file + 1]:
                    has_pawn = False
                    for rank in range(8):
                        check_square = chess.square(check_file, rank)
                        piece = board.piece_at(check_square)
                        if piece and piece.piece_type == chess.PAWN:
                            has_pawn = True
                            break
                    
                    if not has_pawn:
                        # Open file penalty (-15)
                        score += -15 if color == chess.WHITE else 15
            
            # Count enemy pieces attacking king zone
            opponent_color = not color
            attacks = 0
            
            # Define king zone (king position and surrounding squares)
            king_zone = []
            for file_offset in [-1, 0, 1]:
                for rank_offset in [-1, 0, 1]:
                    zone_file = king_file + file_offset
                    zone_rank = king_rank + rank_offset
                    
                    if 0 <= zone_file <= 7 and 0 <= zone_rank <= 7:
                        king_zone.append(chess.square(zone_file, zone_rank))
            
            # Count attacks to king zone
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for piece_square in board.pieces(piece_type, opponent_color):
                    # Check if this piece attacks any square in king zone
                    for zone_square in king_zone:
                        if board.is_attacked_by(opponent_color, zone_square):
                            attacks += 1
                            # Higher penalty for attacks by more valuable pieces
                            attack_value = piece_type * 2  # Knights=6, Bishops=8, Rooks=10, Queens=18
                            score += -attack_value if color == chess.WHITE else attack_value
                            break  # Count each attacking piece only once
        
        return score
    
    def evaluate_center_control(self, board):
        """Evaluate control of the center squares (d4, e4, d5, e5)"""
        score = 0
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        
        for square in center_squares:
            # Check piece occupation
            piece = board.piece_at(square)
            if piece:
                # Bonus for controlling center with a piece
                control_value = 15
                score += control_value if piece.color == chess.WHITE else -control_value
            
            # Check square attacks/control
            if board.is_attacked_by(chess.WHITE, square):
                # Bonus for attacking center with white
                score += 5
            if board.is_attacked_by(chess.BLACK, square):
                # Bonus for attacking center with black
                score -= 5
        
        return score
    
    def evaluate_bishop_pair(self, board):
        """Evaluate bishop pair bonus"""
        score = 0
        
        # Count white bishops
        white_bishops = len(list(board.pieces(chess.BISHOP, chess.WHITE)))
        # Count black bishops
        black_bishops = len(list(board.pieces(chess.BISHOP, chess.BLACK)))
        
        # Bonus for bishop pair
        if white_bishops >= 2:
            score += 50  # Bishop pair bonus
        if black_bishops >= 2:
            score -= 50  # Bishop pair bonus
        
        return score
    
    def evaluate_rooks_on_open_files(self, board):
        """Evaluate bonus for rooks on open or semi-open files"""
        score = 0
        
        # Check each file
        for file_idx in range(8):
            white_pawns = 0
            black_pawns = 0
            
            # Count pawns on this file
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    if piece.color == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            
            # Check if file is open (no pawns) or semi-open (only opponent pawns)
            is_open = white_pawns == 0 and black_pawns == 0
            is_semi_open_white = white_pawns == 0 and black_pawns > 0
            is_semi_open_black = black_pawns == 0 and white_pawns > 0
            
            # Check for rooks on this file
            for rank_idx in range(8):
                square = chess.square(file_idx, rank_idx)
                piece = board.piece_at(square)
                
                if piece and piece.piece_type == chess.ROOK:
                    if is_open:
                        # Bonus for rook on open file
                        score += 25 if piece.color == chess.WHITE else -25
                    elif is_semi_open_white and piece.color == chess.WHITE:
                        # Bonus for white rook on semi-open file
                        score += 15
                    elif is_semi_open_black and piece.color == chess.BLACK:
                        # Bonus for black rook on semi-open file
                        score -= 15
        
        return score
    
    def evaluate_knight_outposts(self, board):
        """Evaluate knight outposts - knights protected by pawns, deep in enemy territory"""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            opponent_color = not color
            
            # Define rank ranges based on color
            middle_ranks = range(3, 7) if color == chess.WHITE else range(1, 5)
            
            for square in board.pieces(chess.KNIGHT, color):
                rank = chess.square_rank(square)
                file = chess.square_file(square)
                
                # Check if knight is in middle ranks
                if rank in middle_ranks:
                    # Check if knight is protected by a friendly pawn
                    is_protected = False
                    
                    # Define pawn protection offsets based on color
                    if color == chess.WHITE:
                        pawn_offsets = [(-1, -1), (1, -1)]  # Diagonal squares behind for white
                    else:
                        pawn_offsets = [(-1, 1), (1, 1)]    # Diagonal squares behind for black
                    
                    for file_offset, rank_offset in pawn_offsets:
                        pawn_file = file + file_offset
                        pawn_rank = rank + rank_offset
                        
                        if 0 <= pawn_file <= 7 and 0 <= pawn_rank <= 7:
                            pawn_square = chess.square(pawn_file, pawn_rank)
                            pawn = board.piece_at(pawn_square)
                            if pawn and pawn.piece_type == chess.PAWN and pawn.color == color:
                                is_protected = True
                                break
                    
                    # Check if square is not attackable by opponent pawns
                    cannot_be_attacked_by_pawns = True
                    
                    if color == chess.WHITE:
                        pawn_attack_offsets = [(-1, 1), (1, 1)]  # Diagonal squares in front for black
                    else:
                        pawn_attack_offsets = [(-1, -1), (1, -1)]  # Diagonal squares in front for white
                    
                    for file_offset, rank_offset in pawn_attack_offsets:
                        attack_file = file + file_offset
                        attack_rank = rank + rank_offset
                        
                        if 0 <= attack_file <= 7 and 0 <= attack_rank <= 7:
                            attack_square = chess.square(attack_file, attack_rank)
                            attacker = board.piece_at(attack_square)
                            if attacker and attacker.piece_type == chess.PAWN and attacker.color == opponent_color:
                                cannot_be_attacked_by_pawns = False
                                break
                    
                    # Bonus for knight outpost
                    if is_protected and cannot_be_attacked_by_pawns:
                        # Higher bonus for advanced knights
                        outpost_rank = rank if color == chess.WHITE else 7 - rank
                        outpost_bonus = 20 + outpost_rank * 5
                        score += outpost_bonus if color == chess.WHITE else -outpost_bonus
        
        return score
    
    def store_transposition(self, board, depth, value, flag, best_move=None):
        """Store a position in the transposition table"""
        # Use Zobrist hash if available, otherwise fallback to FEN
        board_hash = board.fen()
        
        # Store the entry in the transposition table
        self.transposition_table[board_hash] = [depth, value, flag, best_move]
        
        # If table is too large, remove oldest entries
        if len(self.transposition_table) > self.tt_size:
            self.transposition_table.popitem(last=False)
    
    def lookup_transposition(self, board, depth, alpha, beta):
        """Look up a position in the transposition table"""
        # Use Zobrist hash if available, otherwise fallback to FEN
        board_hash = board.fen()
        
        if board_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[board_hash]
            
            # Only use the stored value if it was searched to at least the same depth
            if stored_depth >= depth:
                if stored_flag == 0:  # Exact value
                    return stored_value, stored_move
                elif stored_flag == 1 and stored_value <= alpha:  # Upper bound
                    return alpha, stored_move
                elif stored_flag == 2 and stored_value >= beta:   # Lower bound
                    return beta, stored_move
        
        return None, None
    
    def quiescence_search(self, board, alpha, beta, depth=0):
        """
        Quiescence search to handle the horizon effect
        Only searches captures to reach a "quiet" position
        """
        # Check time before proceeding
        if time.time() - self.start_time > self.time_limit or self.should_stop:
            self.should_stop = True
            return 0
        
        self.nodes_evaluated += 1
        
        # Stand-pat score
        stand_pat = self.evaluate_position(board)
        
        # Beta cutoff
        if stand_pat >= beta:
            return beta
        
        # Update alpha if stand-pat score is better
        if stand_pat > alpha:
            alpha = stand_pat
        
        # Maximum depth for quiescence search
        if depth >= 5:
            return stand_pat
        
        # Generate and sort capture moves
        captures = self.order_moves(board, [], is_quiescence=True)
        
        for move in captures:
            # Skip non-captures in quiescence search
            if not board.is_capture(move):
                continue
                
            board.push(move)
            
            # Skip if we're in check (might lead to zwischenzug)
            if board.is_check():
                score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
            else:
                score = -self.quiescence_search(board, -beta, -alpha, depth + 1)
                
            board.pop()
            
            # Stop if we're out of time
            if self.should_stop:
                return 0
            
            # Beta cutoff
            if score >= beta:
                return beta
            
            # Update alpha if we found a better move
            if score > alpha:
                alpha = score
        
        return alpha
    
    def mvv_lva_score(self, board, move):
        """Score a move based on Most Valuable Victim - Least Valuable Aggressor"""
        if not board.is_capture(move):
            return 0
            
        victim_square = move.to_square
        victim_piece = board.piece_at(victim_square)
        
        # Handle en passant captures
        if board.is_en_passant(move):
            return 100  # Capturing a pawn
            
        # Regular captures
        if victim_piece:
            aggressor_piece = board.piece_at(move.from_square)
            
            # Score = victim value * 10 - aggressor value
            # This prioritizes capturing valuable pieces with less valuable pieces
            return self.piece_values[victim_piece.piece_type] * 10 - self.piece_values[aggressor_piece.piece_type]
        
        return 0
    
    def order_moves(self, board, previous_best_move=None, is_quiescence=False, ply=0):
        """Order moves to improve alpha-beta pruning efficiency"""
        moves = list(board.legal_moves)
        move_scores = []
        
        for move in moves:
            score = 0
            
            # PV move from previous search (highest priority)
            if previous_best_move and move == previous_best_move:
                score += 10000
            
            # Captures (based on MVV-LVA)
            score += self.mvv_lva_score(board, move)
            
            # Killer moves (good non-capturing moves from earlier in the search)
            if self.killer_moves[ply][0] == move:
                score += 900
            elif self.killer_moves[ply][1] == move:
                score += 800
            
            # History heuristic
            move_key = (board.turn, move.from_square, move.to_square)
            if move_key in self.history_table:
                score += min(self.history_table[move_key] // 10, 700)  # Cap history score
            
            # Castling is generally good
            if board.is_castling(move):
                score += 500
            
            # Promotions
            if move.promotion:
                score += 400 + move.promotion
            
            # Check extensions
            if board.gives_check(move):
                score += 300
            
            move_scores.append((move, score))
        
        # Sort moves based on scores (highest first)
        ordered_moves = [move for move, score in sorted(move_scores, key=lambda x: x[1], reverse=True)]
        
        # For quiescence search, only return captures
        if is_quiescence:
            return [move for move in ordered_moves if board.is_capture(move)]
        
        return ordered_moves
    
    def update_killer_moves(self, move, ply):
        """Update the killer moves table with a good non-capturing move"""
        if self.killer_moves[ply][0] != move:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
    
    def update_history_table(self, move, depth, board):
        """Update the history heuristic table"""
        move_key = (board.turn, move.from_square, move.to_square)
        if move_key not in self.history_table:
            self.history_table[move_key] = 0
        self.history_table[move_key] += depth * depth
    
    def negamax_alpha_beta(self, board, depth, alpha, beta, ply=0, null_move_allowed=True):
        """
        Negamax alpha-beta search enhanced with various pruning techniques
        """
        # Check time before proceeding
        if time.time() - self.start_time > self.time_limit or self.should_stop:
            self.should_stop = True
            return 0, None
        
        self.nodes_evaluated += 1
        
        # Check for repetition or game over
        if board.is_repetition(2) or board.is_fifty_moves():
            return 0, None
        
        # Check transposition table
        alpha_orig = alpha
        tt_value, tt_move = self.lookup_transposition(board, depth, alpha, beta)
        if tt_value is not None:
            return tt_value, tt_move
        
        # Base cases: leaf node or terminal position
        if depth == 0 or board.is_game_over():
            # Quiescence search at leaf nodes
            q_score = self.quiescence_search(board, alpha, beta)
            return q_score, None
        
        # Null move pruning
        if null_move_allowed and depth >= 3 and not board.is_check():
            # Check if we have enough material to consider null move
            has_major_pieces = False
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece and piece.color == board.turn and piece.piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                    has_major_pieces = True
                    break
            
            if has_major_pieces:
                board.push(chess.Move.null())
                null_score, _ = self.negamax_alpha_beta(board, depth - 1 - self.R, -beta, -beta + 1, ply + 1, False)
                board.pop()
                
                null_score = -null_score
                
                if null_score >= beta:
                    return beta, None  # Null move cutoff
        
        best_move = None
        best_score = float('-inf')
        
        # Get ordered moves
        ordered_moves = self.order_moves(board, tt_move, False, ply)
        
        # Internal iterative deepening if no tt_move and we're at sufficient depth
        if not tt_move and depth >= 4:
            _, iid_move = self.negamax_alpha_beta(board, depth - 2, alpha, beta, ply, False)
            if iid_move:
                # Reorder moves to place iid_move first
                ordered_moves.remove(iid_move)
                ordered_moves.insert(0, iid_move)
        
        moves_searched = 0
        for move in ordered_moves:
            board.push(move)
            
            # Late Move Reduction (LMR) - reduce search depth for later moves
            if moves_searched >= self.lmr_full_depth_moves and depth >= self.lmr_reduction_limit and not board.is_check() and not move.promotion and not board.is_capture(move):
                # Reduced depth search
                reduction = 1 if moves_searched >= self.lmr_full_depth_moves + 4 else 0
                score, _ = self.negamax_alpha_beta(board, depth - 1 - reduction, -beta, -alpha, ply + 1, null_move_allowed)
                score = -score
                
                # Verify with full depth search if score is promising
                if score > alpha and reduction > 0:
                    score, _ = self.negamax_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, null_move_allowed)
                    score = -score
            else:
                # Full depth search
                score, _ = self.negamax_alpha_beta(board, depth - 1, -beta, -alpha, ply + 1, null_move_allowed)
                score = -score
            
            board.pop()
            
            moves_searched += 1
            
            # Stop search if we're out of time
            if self.should_stop:
                return 0, None
            
            # Update best score and move
            if score > best_score:
                best_score = score
                best_move = move
                
                # Update alpha
                if score > alpha:
                    alpha = score
                    
                    # Beta cutoff
                    if alpha >= beta:
                        if not board.is_capture(move):
                            # Update killer moves for good non-capturing moves
                            self.update_killer_moves(move, ply)
                            # Update history table
                            self.update_history_table(move, depth, board)
                        
                        # Store in transposition table as a lower bound
                        self.store_transposition(board, depth, beta, 2, move)
                        return beta, move
        
        # Store the result in the transposition table
        if best_score <= alpha_orig:
            # Upper bound
            self.store_transposition(board, depth, best_score, 1, best_move)
        elif best_score >= beta:
            # Lower bound
            self.store_transposition(board, depth, best_score, 2, best_move)
        else:
            # Exact value
            self.store_transposition(board, depth, best_score, 0, best_move)
        
        return best_score, best_move
    
    def check_opening_book(self, board):
        """Check if the current position is in the opening book"""
        fen = board.fen().split(' ')[0] + ' ' + ('w' if board.turn == chess.WHITE else 'b')
        
        # Try to match position
        for book_fen, book_moves in self.opening_book.items():
            if book_fen.startswith(fen):
                # Return a random move from the book
                if book_moves:
                    move_uci = random.choice(book_moves)
                    try:
                        return chess.Move.from_uci(move_uci)
                    except ValueError:
                        continue
        
        return None  # Not in book
    
    def get_best_move(self, board, max_depth=4):
        """Find the best move using iterative deepening alpha-beta search"""
        self.start_time = time.time()
        self.nodes_evaluated = 0
        self.should_stop = False
        
        # Check opening book first
        book_move = self.check_opening_book(board)
        if book_move:
            return book_move
        
        # Ensure max_depth is at least 1
        max_depth = max(1, max_depth)
        
        best_move = None
        best_move_score = float('-inf') if board.turn == chess.WHITE else float('inf')
        
        try:
            # Iterative deepening
            for depth in range(1, max_depth + 1):
                if time.time() - self.start_time > 0.75 * self.time_limit:
                    # If we've used 75% of our time, stop the search
                    break
                    
                score, current_best_move = self.negamax_alpha_beta(
                    board, 
                    depth, 
                    float('-inf'), 
                    float('inf'), 
                    0, 
                    True
                )
                
                # If we're out of time, don't update the best move
                if self.should_stop and depth > 1:
                    break
                
                # Update best move if we found one
                if current_best_move:
                    best_move = current_best_move
                    best_move_score = score
                
                # Aspiration window for next iteration
                if depth >= 3 and not self.should_stop:
                    # Prepare narrower window for next iteration
                    alpha = score - 50
                    beta = score + 50
                    
                    # Adjust window if it's too narrow
                    if alpha <= -20000:
                        alpha = float('-inf')
                    if beta >= 20000:
                        beta = float('inf')
                
                # If we're out of time or stopped, break
                if self.should_stop or time.time() - self.start_time > 0.9 * self.time_limit:
                    break
                    
        except Exception as e:
            # If any error occurs during search, we'll fall back to a fallback engine
            print(f"Error in search: {e}")
            best_move = None
        
        # If we didn't find a move or encountered an error, try the first legal move
        if not best_move:
            try:
                best_move = next(iter(board.legal_moves))
            except StopIteration:
                # No legal moves
                return None
        
        return best_move
    
    def play(self):
        """
        Return the move played by the AI.
        Falls back to GreedyExplorationAI if Stockfish takes too long.
        """
        board = self.game.board
        start_time = time.time()
        
        try:
            # First attempt with full Stockfish (enhanced) algorithm
            # Adjust depth based on game phase and time constraints
            piece_count = sum(1 for _ in board.piece_map())
            
            # Determine appropriate depth based on game phase
            if piece_count > 28:  # Opening
                max_depth = 4
            elif piece_count < 12:  # Endgame
                max_depth = 5
            else:  # Middlegame
                max_depth = 4
            
            move = self.get_best_move(board, max_depth)
            
            # Verify move is legal (should always be, but double-check)
            if move and move in board.legal_moves:
                return move
                
            # If we get here, something went wrong with our search
            raise Exception("Move not legal or not found")
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ScratchFishAI error: {e}, elapsed: {elapsed}s, falling back to GreedyExplorationAI")
            
            # Fall back to GreedyExplorationAI
            if self.greedy_exploration is None:
                self.greedy_exploration = GreedyExplorationAI()
                self.greedy_exploration.game = self.game
                self.greedy_exploration.color = self.color
            
            try:
                return self.greedy_exploration.play()
            except Exception as e2:
                print(f"GreedyExplorationAI error: {e2}, falling back to GreedyAI")
                
                # Final fallback to basic GreedyAI
                if self.greedy is None:
                    self.greedy = GreedyAI()
                    self.greedy.game = self.game
                    self.greedy.color = self.color
                
                return self.greedy.play()