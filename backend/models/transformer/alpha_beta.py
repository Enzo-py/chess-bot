import chess
import torch
import numpy as np
import torch.nn.functional as F

from src.chess.simulation import Simulation
from models.deep_engine import DeepEngine, Game
from models.transformer.transformer import ChessTransformerEncoder, BoardEvaluator, GenerativeHead, Decoder


class AlphaBetaTransformerEngine(DeepEngine):
    """
    Alpha-Beta search engine that uses a transformer model for board evaluation.
    Similar to the CNN version but using transformer components for more advanced pattern recognition.
    """
    __author__ = "Luis Wiedmann"
    __description__ = "AlphaBeta search with Transformer evaluation using DeepEngine framework"
    __weights__ = "TransformerScore"

    def __init__(self, max_depth=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_depth = max_depth

        # Use the transformer components
        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessTransformerEncoder())
        self.set(head_name="decoder", head=Decoder())

    def evaluate_board(self, game):
        """
        Use the transformer encoder and board evaluation head to score a game.
        """
        self.game = game
        features = self.predict(head="encoder")  # get features from encoder
        probs = self.predict(head="board_evaluation")  # get win probabilities from board evaluation head
        return probs[1] - probs[0]  # white win prob - black win prob

    def alphabeta(self, board, depth, alpha, beta, maximizing):
        """
        Alpha-Beta search, guided by the transformer evaluation.
        """
        if depth == 0 or board.is_game_over():
            game = self.create_game_from_board(board)
            return self.evaluate_board(game)

        legal_moves = list(board.legal_moves)

        if maximizing:
            max_eval = -float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def choose_move(self, game):
        """
        Choose the best move for the current game using Alpha-Beta search.
        """
        self.game = game
        board = game.board

        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

        # For each legal move, evaluate the position after that move
        for move in board.legal_moves:
            board.push(move)
            # USE TRANSFORMER FOR CHOOSING THE MOVE
            #TAKE THE transformer functions for making this work
            value = self.alphabeta(board, self.max_depth - 1, -float('inf'), float('inf'), board.turn == chess.BLACK)
            board.pop()

            if (board.turn == chess.WHITE and value > best_value) or (board.turn == chess.BLACK and value < best_value):
                best_value = value
                best_move = move

        return best_move

    @staticmethod
    def create_game_from_board(board):
        """
        Wrap a chess.Board into a Game object expected by DeepEngine.
        """
        game = Game()
        game.board = board.copy()
        return game

    def play(self):
        """
        Required DeepEngine method - pick the move using Alpha-Beta.
        """
        if not self.is_setup:
            raise ValueError("Model not setup.")
        return self.choose_move(self.game)


class AlphaBetaTransformerEngineWithSorting(AlphaBetaTransformerEngine):
    """
    Enhanced version of AlphaBetaTransformerEngine with move ordering optimization.
    
    This version sorts moves before evaluation to improve alpha-beta pruning efficiency,
    using a quick heuristic for ordering (captures and promotions first).
    """
    __author__ = "Luis Wiedmann" 
    __description__ = "AlphaBeta search with Transformer evaluation and move ordering"
    
    def __init__(self, max_depth=3, *args, **kwargs):
        super().__init__(max_depth, *args, **kwargs)
    
    def move_value(self, board, move):
        """
        Assign a preliminary value to a move for ordering purposes.
        Higher values for moves that are likely to be good.
        """
        # Prioritize captures, with MVV-LVA (Most Valuable Victim - Least Valuable Aggressor)
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        
        value = 0
        
        # Check if move is a capture
        if board.is_capture(move):
            # Get the captured piece
            to_square = move.to_square
            captured_piece = board.piece_at(to_square)
            if captured_piece:
                value += 10 * piece_values.get(captured_piece.piece_type, 0)
            
            # Subtract value of capturing piece (prefer capturing with less valuable pieces)
            from_square = move.from_square
            capturing_piece = board.piece_at(from_square)
            if capturing_piece:
                value -= piece_values.get(capturing_piece.piece_type, 0)
        
        # Prioritize promotions
        if move.promotion:
            value += 9  # Queen promotion value
            
        # Prioritize checks (optional, might be expensive to calculate)
        board.push(move)
        if board.is_check():
            value += 3
        board.pop()
        
        return value
    
    def alphabeta(self, board, depth, alpha, beta, maximizing):
        """
        Enhanced Alpha-Beta search with move ordering for better pruning.
        """
        if depth == 0 or board.is_game_over():
            game = self.create_game_from_board(board)
            return self.evaluate_board(game)

        # Get legal moves and sort them
        legal_moves = list(board.legal_moves)
        scored_moves = [(move, self.move_value(board, move)) for move in legal_moves]
        
        # Sort moves: descending for maximizing player, ascending for minimizing
        if maximizing:
            scored_moves.sort(key=lambda x: x[1], reverse=True)
        else:
            scored_moves.sort(key=lambda x: x[1])
            
        sorted_moves = [move for move, _ in scored_moves]

        if maximizing:
            max_eval = -float('inf')
            for move in sorted_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in sorted_moves:
                board.push(move)
                eval = self.alphabeta(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
            
    def choose_move(self, game):
        """
        Choose the best move with move ordering for the first level.
        """
        self.game = game
        board = game.board

        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

        # Score and sort all legal moves
        legal_moves = list(board.legal_moves)
        scored_moves = [(move, self.move_value(board, move)) for move in legal_moves]
        
        if board.turn == chess.WHITE:
            scored_moves.sort(key=lambda x: x[1], reverse=True)  # Descending for white
        else:
            scored_moves.sort(key=lambda x: x[1])  # Ascending for black
            
        sorted_moves = [move for move, _ in scored_moves]

        # Evaluate each move
        for move in sorted_moves:
            board.push(move)
            value = self.alphabeta(board, self.max_depth - 1, -float('inf'), float('inf'), board.turn == chess.BLACK)
            board.pop()

            if (board.turn == chess.WHITE and value > best_value) or (board.turn == chess.BLACK and value < best_value):
                best_value = value
                best_move = move

        return best_move 