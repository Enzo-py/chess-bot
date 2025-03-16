import sys
import os
import torch
import chess

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.models.transformer.transformer_score import TransformerScore
from src.chess.game import Game
from src.utils.console import Style

def main():
    # Create an instance of the TransformerScore model
    model = TransformerScore()
    print(f"{Style.GREEN}Created TransformerScore model{Style.RESET}")
    
    # Create a chess game in the starting position
    board = chess.Board()
    game = Game(board)
    
    # Evaluate the current position
    score = model.evaluate(game)
    print(f"{Style.BLUE}Board evaluation (win probability for white): {score:.4f}{Style.RESET}")
    
    # Get the best move using the model
    try:
        best_move = model.play(game=game, temperature=0.0)
        print(f"{Style.GREEN}Best move according to transformer: {best_move.uci()}{Style.RESET}")
    except Exception as e:
        print(f"{Style.RED}Error getting best move: {e}{Style.RESET}")
    
    # Print model structure
    print(f"\n{Style.BLUE}Model structure:{Style.RESET}")
    print(model.manifest)
    
    # Example of saving the model
    save_path = "models/saves/transformer_score.pth"
    model.save(save_path)
    print(f"{Style.GREEN}Model saved to {save_path}{Style.RESET}")
    
if __name__ == "__main__":
    main() 