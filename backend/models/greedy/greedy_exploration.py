from models.engine import Engine
from .greedy_ai import GreedyAI
from src.chess.simulation import Simulation

import chess
import numpy as np

class GreedyExplorationAI(Engine):
    """
    Optimized Greedy AI that plays as strongly as possible with tree exploration.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "Optimized Greedy AI that plays as strongly as possible with tree exploration."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy = None

        self.exploration_size = 20
        self.exploration_depth = 3
        self.exploration_sample = 100
        self.choice_exploration = 3

        assert self.exploration_sample % 2 == 0, "Exploration sample must be even"

    def play(self) -> chess.Move:
        # Initialize GreedyAI only once
        if self.greedy is None: 
            self.greedy = GreedyAI()
            self.greedy.game = self.game
            self.greedy.color = self.color
        
        # Get topN candidate moves
        top_moves = self.greedy.play(topN=self.exploration_size)
        if not top_moves:
            return None  # No legal moves available

        move_scores = np.zeros(len(top_moves))
        # Explore each move using simulations
        for i, move in enumerate(top_moves):
            with Simulation(self.game) as sm:
                scores = np.zeros(self.exploration_sample)

                for j in range(self.exploration_sample):
                    sm.game.move(move)
                    sm.run(engine=GreedyAI, depth=self.exploration_depth, play_args={"topN": self.choice_exploration})
                    
                    # Early stopping if game is over
                    if sm.game.is_game_over():
                        if sm.game.checkmate != self.color:
                            scores[j] = 1e6  # Winning move
                            return move
                        elif sm.game.checkmate == self.color:
                            scores[j] = -1e6  # Losing move
                            break
                        else:
                            scores[j] = 0  # Stalemate
                        break
                    else:
                        scores[j] = self.get_score(sm.game)
                    
                    if move.to_square in sm.game.BB_EDGE:
                        scores[j] -= 100  # Penalty for moving to the edge
                    
                    # if move is a castling move
                    if move.from_square == 4 and move.to_square == 6:
                        scores[j] += 200
                    if move.from_square == 4 and move.to_square == 2:
                        scores[j] += 200
                    if move.from_square == 60 and move.to_square == 62:
                        scores[j] += 200
                    if move.from_square == 60 and move.to_square == 58:
                        scores[j] += 200
                    
                    sm.reset()

                # Store the mean score for this move
                move_scores[i] = np.mean(scores)

        # Return the best move based on the highest score
        argmax = np.argmax(move_scores)
        return top_moves[argmax]

    def get_score(self, game):
        """
        Evaluate the board position for self.color.
        - Material count (weighted)
        - Piece activity (mobility)
        - King safety (castling, pawn shield)
        - Center control
        """
        board = game.board  # Reference to the board
        color = self.color       # AI's color
        score = 0

        # Piece values with small bonuses
        PIECE_VALUES = {
            chess.PAWN: 15, chess.KNIGHT: 32, chess.BISHOP: 33,
            chess.ROOK: 50, chess.QUEEN: 90, chess.KING: 0
        }

        # 1️⃣ Material count
        for piece_type, value in PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, color))
            score -= value * len(board.pieces(piece_type, not color))

        score **3

        # 2️⃣ Piece activity (mobility)
        legal_moves = list(board.legal_moves)

        # 3️⃣ Center control (bonus for pawns and knights in the center)
        center_squares = chess.SquareSet(chess.BB_CENTER)
        for move in legal_moves:
            from_square, to_square = move.from_square, move.to_square
            piece = board.piece_at(from_square)
            if piece.piece_type == chess.PAWN:
                if to_square in center_squares:
                    score += 5
                if to_square in chess.SquareSet(chess.BB_RANK_8 | chess.BB_RANK_1):
                    score += 30
                if from_square in chess.SquareSet(chess.BB_RANK_2 | chess.BB_RANK_7):
                    score += 1
                if to_square in chess.SquareSet(chess.BB_RANK_3 | chess.BB_RANK_6):
                    score += 1
            elif piece.piece_type != chess.KING:
                if to_square in center_squares:
                    score += 3

        # if ennemy king is in the center
        ennemy_king = board.king(not color)
        if ennemy_king in center_squares:
            score += 100

        # if ennemy king don't have a lot of moves
        ennemy_king_box = self.game.find_piece_box(chess.KING, not color)
        moves = len(self.game.get_possible_moves(ennemy_king_box))
        score -= moves * 2
   
        # 4️⃣ King safety (penalty for unsafe kings)
        king_square = board.king(color)
        if king_square:
            if board.has_kingside_castling_rights(color) or board.has_queenside_castling_rights(color):
                score += 20  # Bonus for castling rights 
            else:
                score -= 20  # Slight penalty if king can't castle

            # Bonus for pawns protecting the king
            pawn_shield_bonus = sum(
                10 for sq in chess.SquareSet(chess.BB_PAWN_ATTACKS[color][king_square]) 
                if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN
            )
            score += pawn_shield_bonus

        return score
