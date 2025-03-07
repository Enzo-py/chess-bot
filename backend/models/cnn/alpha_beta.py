import chess
import torch
import numpy as np
import torch.nn.functional as F

from src.chess.simulation import Simulation
from models.deep_engine import DeepEngine, Game
from models.cnn.cnn_score import BoardEvaluator, ChessEmbedding


class AlphaBetaDeepEngine(DeepEngine):
    __author__ = "Yann Baglin-Bunod"
    __description__ = "AlphaBeta search with CNN evaluation using DeepEngine framework"

    def __init__(self, max_depth=3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_depth = max_depth

        # Use the same CNN components already available in your project
        self.set(head_name="encoder", head=ChessEmbedding())
        self.set(head_name="board_evaluation", head=BoardEvaluator())

    def evaluate_board(self, game):
        """
        Use the CNN encoder and board evaluation head to score a game.
        """
        self.game = game
        features = self.predict(head="encoder")  # get features from encoder
        probs = self.predict(head="board_evaluation")  # get win probabilities from board evaluation head
        return probs[1] - probs[0]  # white win prob - black win prob

    def alphabeta(self, board, depth, alpha, beta, maximizing):
        """
        Alpha-Beta search, guided by the CNN evaluation.
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

        for move in board.legal_moves:
            board.push(move)
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
        This assumes you have some way of converting board -> Game.
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