import numpy as np
from src.chess.simulation import Simulation
from models.deep_engine import DeepEngine
from models.cnn.cnn_score import BoardEvaluator, GenerativeHead, ChessEmbedding
import torch
import torch.nn.functional as F
import chess

class TreeSearchCNN(DeepEngine):
    """
    CNN-based AI that scores the board state.
    """
    __author__ = "Enzo Pinchon"
    __description__ = "CNN-based AI that scores the board state."
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessEmbedding())

        self.top_proba = 0.35
        self.temperature = 1

        # Exploration parameters:
        self.exploration_size = 5         # Number of top candidate moves
        self.exploration_depth = 14         # Simulation depth
        self.exploration_sample = 200      # Number of simulation samples per candidate move
        self.confidence = 50               # Confidence parameter for Beta distribution
        # Ensure even number of samples (if needed by your logic)
        assert self.exploration_sample % 2 == 0, "Exploration sample must be even"

    def choose_move(self) -> chess.Move:
        legal_moves = list(self.game.board.legal_moves)
        logits = [
            self.predict(head="generation")[self.encode_move(move, as_int=True)]
            for move in legal_moves
        ]
        probs = F.softmax(torch.tensor(logits), dim=0).detach().numpy() ** (1 / self.temperature)
        probs = np.sort(probs)[::-1]
        total_proba = 0
        possibles_moves = []

        idx = 0
        while total_proba < self.top_proba and idx < len(probs):
            move = legal_moves[idx]
            total_proba += probs[idx]
            possibles_moves.append(move)
            idx += 1

        if len(possibles_moves) == 0:
            return None
        
        probs = probs[:idx]
        total = sum(probs)
        probs = [p / total for p in probs]
        return np.random.choice(possibles_moves, p=probs)

    def play(self) -> chess.Move:
        if not self.is_setup: raise ValueError("Model not setup.")
        return self.choose_move()
