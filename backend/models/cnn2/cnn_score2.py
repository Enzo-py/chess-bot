import numpy as np
from models.deep_engine import DeepEngine
from models.cnn.cnn_score import BoardEvaluator, GenerativeHead, ChessEmbedding, Decoder

from src.chess.game import Game
import os
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess.syzygy 

class SyzygyEvaluator:
    def __init__(self, tb_dir):
        self.tb_dir = tb_dir
        self.tablebase = chess.syzygy.open_tablebase(tb_dir)

    def evaluate(self, board: chess.Board):
        if len(board.piece_map()) <= 7:
            try:
                wdl = self.tablebase.probe_wdl(board)
                if wdl > 0:
                    return 1.0
                elif wdl == 0:
                    return 0.5
                else:
                    return 0.0
            except chess.syzygy.MissingTableError:
                return None
        return None

class CNNScore2(DeepEngine):
    """
    CNN-based AI that scores the board state.
    """

    __author__ = "Matt"
    __description__ = "CNN-based AI that scores the board state."
    __weights__ = "CNNScore"
        
    def __init__(self):
        super().__init__()

        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessEmbedding())
        tb_path = os.path.join( "data", "syzygy_tablebases")
        self.syzygy_evaluator = SyzygyEvaluator(tb_path)

    def play(self) -> chess.Move:
        """
        If Syzygy tablebases are available, returns the optimal move.
        else use best CNN Move.
        """
        board = self.game.board
        # Check if we can use Syzygy tablebases (position has 7 or fewer pieces)
        if len(board.piece_map()) <= 7:
            try:
                moves = list(board.legal_moves)
                best_move = None
                best_score = -float('inf')
                
                for move in moves:
                    board_copy = board.copy()
                    board_copy.push(move)
                    
                    if not board_copy.is_game_over():
                        score = self.syzygy_evaluator.evaluate(board_copy)
                        
                        if score is not None:
                            opponent_score = 1.0 - score
                            
                            if opponent_score > best_score:
                                best_score = opponent_score
                                best_move = move
                
                # If we found a tablebase-optimal move, return it
                if best_move:
                    print("Best move found")
                    return best_move
                    
            except Exception as e:
                # Fall back to CNN on any error
                pass
        
        scores = self.predict()
        legal_moves = list(self.game.board.legal_moves)
        scores = [scores[self.encode_move(move, as_int=True)] for move in legal_moves]
        return legal_moves[scores.index(max(scores))]
  
