from models.engine import Engine
import chess
import random
from typing import Dict, List, Optional
import json
import chess.pgn
import zstandard as zstd
import io
import os

class TDLearningAI(Engine):
    """
    Chess AI using TD-learning.
    
    This implementation uses TD(0) to learn a simple state value function.
    During self-play the agent updates its value table based on the difference
    between successive predictions.
    """
    
    __author__ = "Matt"
    __description__ = "TD-learning AI for chess using TD(0) updates."
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.score_function = type("DummyScoreFunction", (), {"train": lambda self: None})()

        self.alpha = 0.3                # Learning rate
        self.gamma = 0.95               # Discount factor
        self.epsilon = 0.5              # Exploration rate (epsilon-greedy)
        
        value_table_path = "./models/saves/td_value_table_final.json"
        # os.makedirs("backend/models/saves", exist_ok=True)
        # print(f"Directory exists: {os.path.isdir('backend/models/saves')}")
        print(f"Working directory: {os.getcwd()}")
        if os.path.exists(value_table_path):
            print(f"Loading existing value table from {value_table_path}")
            self.load_value_table(value_table_path)
            print(f"Loaded {len(self.value_table)} positions")
        else:
            self.value_table = {}
            print("No existing value table found. Starting with empty table.")

        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }

    def get_value(self, board: chess.Board) -> float:
        """Return current value estimate for the board position (FEN).
           Defaults to 0.5 meaning a roughly balanced position."""
        fen = board.fen()
        return self.value_table.get(fen, 0.5)
    
    def update_value(self, board: chess.Board, target: float):
        """TD update for a given board position."""
        fen = board.fen()
        current = self.value_table.get(fen, 0.5)
        self.value_table[fen] = current + self.alpha * (target - current)
    
    def choose_epsilon_greedy_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Choose a move using an epsilon-greedy strategy based on current value estimates."""
        moves = list(board.legal_moves)
        if not moves:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(moves)
        
        best_move = None
        best_value = float('-inf')
        for move in moves:
            next_board = board.copy()
            next_board.push(move)
            value = self.get_value(next_board)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move if best_move is not None else random.choice(moves)
    
    def play(self) -> chess.Move:
        """
        Return the move played by the TD-learning AI.
        The move is selected using greedy policy on the current value estimates.
        """
        actions = list(self.game.board.legal_moves)
        if not actions:
            return None
        
        best_move = None
        best_value = float('-inf')
        for move in actions:
            next_board = self.game.board.copy()
            next_board.push(move)
            value = self.get_value(next_board)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
    
    
    def learn(self, num_games:int = 100, pgn_zst_path = "backend/data/lichess_db_standard_rated_2015-05.pgn.zst"):
        """
        Train the TD-learning agent via self-play.
        
        Runs the specified number of episodes updating the value table with TD(0).
        """
         # Open the compressed file
        with open(pgn_zst_path, 'rb') as compressed_file:
            # Create a zstandard decompressor
            dctx = zstd.ZstdDecompressor()
            # Create a stream reader
            reader = dctx.stream_reader(compressed_file)
            # Create a text IO wrapper
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            games_processed = 0
            while games_processed < num_games:
                # Read a game from the PGN
                game = chess.pgn.read_game(text_stream)
                if game is None:  # End of file
                    break
                    
                # Get game result
                result = game.headers.get("Result", "*")
                final_reward = 0.5  # Default to draw
                if result == "1-0":
                    final_reward = 1.0  # White win
                elif result == "0-1":
                    final_reward = 0.0  # Black win
                
                # Play through the game moves and collect board positions
                board = game.board()
                trajectory = []
                
                for move in game.mainline_moves():
                    trajectory.append((board.fen(), board.turn))
                    board.push(move)
                
                # Apply TD learning on the game trajectory
                # For white's perspective (1.0 is win, 0.0 is loss)
                target = final_reward
                for state, turn in reversed(trajectory):
                    # Adjust target based on whose turn it was
                    adjusted_target = target if turn == chess.WHITE else 1.0 - target
                    
                    # Get current value and update
                    current = self.value_table.get(state, 0.5)
                    self.value_table[state] = current + self.alpha * (adjusted_target - current)
                    
                    # Update target for next state
                    target = self.gamma * target + (1 - self.gamma) * current
                
                games_processed += 1
                
                # Optional: Print progress
                if games_processed % 100 == 0:
                    print(f"Processed {games_processed} games")
            
            print(f"Training complete. Processed {games_processed} games.")
            print(f"Value table size: {len(self.value_table)} positions")
        
    def save_value_table(self, filepath: str):
        """Save the value table to a file."""
        with open(filepath, 'w') as f:
            json.dump(self.value_table, f)

    def load_value_table(self, filepath: str):
        """Load the value table from a file."""
        with open(filepath, 'r') as f:
            self.value_table = json.load(f)

    def evaluate_material(self, board: chess.Board) -> float:
        """Evaluate board based on material balance and normalize to [0, 1]."""
        material_score = 0
        for piece_type, value in self.piece_values.items():
            material_score += (
                len(board.pieces(piece_type, chess.WHITE)) * value -
                len(board.pieces(piece_type, chess.BLACK)) * value
            )
        normalized_score = 0.5 + material_score / (self.piece_values[chess.QUEEN] * 4)
        return max(0.0, min(1.0, normalized_score))
    
    def _train_board_evaluation(self, epochs, batch_size, games, labels, loader=None):
        self.learn(num_episodes=epochs)
        return {"epochs": epochs, "batch_size": batch_size, "value_table_size": len(self.value_table)}