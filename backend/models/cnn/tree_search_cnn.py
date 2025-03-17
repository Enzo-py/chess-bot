import numpy as np
from src.chess.simulation import Simulation
from models.deep_engine import DeepEngine
from models.cnn.cnn_score import BoardEvaluator, GenerativeHead, ChessEmbedding, Decoder
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
        self.set(head_name="decoder", head=Decoder())

        self.top_proba = 0.35
        self.temperature = 1
        self.shallow = False

        # Exploration parameters:
        self.exploration_size = 5         # Number of top candidate moves
        self.exploration_depth = 2         # Simulation depth
        self.exploration_sample = 10      # Number of simulation samples per candidate move
        self.confidence = 50               # Confidence parameter for Beta distribution
        # Ensure even number of samples (if needed by your logic)
        assert self.exploration_sample % 2 == 0, "Exploration sample must be even"

    def choose_move(self, get_probs=False) -> chess.Move:
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

        if get_probs: return probs, possibles_moves
        return np.random.choice(possibles_moves, p=probs)

    def play(self) -> chess.Move:
        if not self.is_setup: raise ValueError("Model not setup.")
        if self.shallow: return self.choose_move()

        probs, possibles_moves = self.choose_move(get_probs=True)
        move_score = np.zeros(len(list(possibles_moves)))
        
        if len(possibles_moves) == 0: return None
        if len(possibles_moves) == 1: return list(possibles_moves)[0]
        
        with Simulation(self.game) as sm:
            for i, move in enumerate(possibles_moves):
                print(f"Move {i + 1}/{len(list(possibles_moves))}")
                sm.game.move(move)
                sm.run(engine=TreeSearchCNN, depth=-1)
                move_score[i] = -1 if self.game.checkmate == self.color else (1 if self.game.checkmate == (not self.color) else 0)
                sm.reset()
                move_score[i] *= probs[i]

        return list(possibles_moves)[np.argmax(move_score)]
