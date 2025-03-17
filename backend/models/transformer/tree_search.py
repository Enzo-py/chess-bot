import numpy as np
from src.chess.simulation import Simulation
from models.deep_engine import DeepEngine
from models.transformer.transformer import BoardEvaluator, GenerativeHead, ChessTransformerEncoder, Decoder
import torch
import torch.nn.functional as F
import chess

class TreeSearchTransformer(DeepEngine):
    """
    Transformer-based AI that uses tree search with board evaluation.
    """
    __author__ = "Luis Wiedmann"
    __description__ = "Transformer-based AI that uses tree search with board evaluation."
    __weights__ = "TransformerScore"
 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessTransformerEncoder())
        self.set(head_name="decoder", head=Decoder())

        # Configuration parameters
        self.top_proba = 0.35  # Consider moves that make up top 35% of probability mass
        self.temperature = 1.1  # Temperature for softmax, higher = more exploration
        self.shallow = False    # If True, don't use tree search, just pick from probabilities

        # Exploration parameters:
        self.exploration_depth = 3      # Simulation depth for look-ahead
        self.exploration_sample = 100   # Number of simulation samples per candidate move
        self.confidence = 55            # Confidence parameter for Beta distribution
        
        # Ensure even number of samples (for balanced simulation)
        assert self.exploration_sample % 2 == 0, "Exploration sample must be even"

    def choose_move(self, get_probs=False) -> chess.Move:
        """
        Choose a move based on the generative head's probabilities.
        
        Args:
            get_probs: If True, return probabilities and possible moves as well
            
        Returns:
            A chess move, or (probs, moves) if get_probs is True
        """
        legal_moves = list(self.game.board.legal_moves)
        
        # Get logits for each legal move from the generative head
        logits = [
            self.predict(head="generation")[self.encode_move(move, as_int=True)]
            for move in legal_moves
        ]
        
        # Apply temperature and convert to probabilities
        probs = F.softmax(torch.tensor(logits), dim=0).detach().numpy() ** (1 / self.temperature)
        
        # Sort moves by probability (descending)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]
        sorted_moves = [legal_moves[i] for i in sorted_indices]
        
        # Select moves until we reach the top_proba threshold
        total_proba = 0
        possible_moves = []
        selected_probs = []
        
        idx = 0
        while total_proba < self.top_proba and idx < len(sorted_probs):
            move = sorted_moves[idx]
            prob = sorted_probs[idx]
            total_proba += prob
            possible_moves.append(move)
            selected_probs.append(prob)
            idx += 1

        if len(possible_moves) == 0:
            return None
        
        # Normalize the selected probabilities
        total = sum(selected_probs)
        normalized_probs = [p / total for p in selected_probs]

        if get_probs:
            return normalized_probs, possible_moves
            
        # Choose a move based on the normalized probabilities
        return np.random.choice(possible_moves, p=normalized_probs)

    def play(self) -> chess.Move:
        """
        Play a move using tree search to look ahead.
        
        If shallow=True, simply choose based on move probabilities.
        Otherwise, use tree search to evaluate possible futures.
        
        Returns:
            A chess move
        """
        if not self.is_setup:
            raise ValueError("Model not setup.")
            
        # If shallow mode, just pick based on probabilities without tree search
        if self.shallow:
            return self.choose_move()

        # Get the top moves and their probabilities
        probs, possible_moves = self.choose_move(get_probs=True)
        
        # Edge cases
        if len(possible_moves) == 0:
            return None
        if len(possible_moves) == 1:
            return possible_moves[0]
        
        # Initialize scores for each move
        move_score = np.zeros(len(possible_moves))
        
        # Run simulations to evaluate each move
        with Simulation(self.game) as sm:
            for i, move in enumerate(possible_moves):
                print(f"Evaluating move {i + 1}/{len(possible_moves)}: {move.uci()}")
                
                # Make the move in the simulation
                sm.game.move(move)
                
                # Run a simulation with this move to see how it plays out
                sm.run(engine=TreeSearchTransformer, depth=self.exploration_depth)
                
                # Score based on outcome (weighted by the move's prior probability)
                # -1 if we're checkmated, +1 if opponent is checkmated, 0 otherwise
                if self.game.checkmate == self.color:
                    move_score[i] = -300  # We lost
                elif self.game.checkmate == (not self.color):
                    move_score[i] = 600  # We won
                else:
                    move_score[i] = 0   # Draw or ongoing
                    
                # Weight the score by the move's prior probability
                move_score[i] *= probs[i]
                
                # Reset for next move simulation
                sm.reset()

        # Choose the move with the highest score
        best_move_idx = np.argmax(move_score)
        return possible_moves[best_move_idx]
        
    def evaluate_position(self):
        """
        Evaluate the current position using the board_evaluation head.
        
        Returns:
            A score between -1 and 1, where positive favors white
        """
        probs = self.predict(head="board_evaluation")
        return probs[1] - probs[0]  # white probability - black probability 