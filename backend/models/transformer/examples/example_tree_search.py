import sys
import os
import time
import chess

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.transformer.tree_search import TreeSearchTransformer
from src.chess.game import Game
from src.utils.console import Style

def display_board(board):
    """Display the chess board in a readable format."""
    print("\n  " + "-" * 17)
    for rank in range(7, -1, -1):
        print(f"{rank + 1} |", end="")
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece:
                symbol = piece.symbol()
                print(f" {symbol} ", end="")
            else:
                print(" . ", end="")
        print("|")
    print("  " + "-" * 17)
    print("    a  b  c  d  e  f  g  h")
    print(f"Turn: {'White' if board.turn else 'Black'}")

def main():
    # Create a chess game in the starting position
    board = chess.Board()
    game = Game(board)
    
    print(f"{Style.GREEN}Chess game initialized{Style.RESET}")
    display_board(board)
    
    # Create an instance of the TreeSearchTransformer
    # Note: In a real scenario, you would typically use a pre-trained model
    engine = TreeSearchTransformer()
    print(f"{Style.GREEN}Created TreeSearchTransformer{Style.RESET}")
    
    # Configure the engine for demonstration (using smaller values for faster execution)
    engine.exploration_depth = 2
    engine.exploration_sample = 10
    engine.temperature = 1.2  # Slightly higher temperature for more exploration
    
    # First, try with shallow search (just generative probabilities, no tree search)
    engine.shallow = True
    print(f"\n{Style.BLUE}Using shallow search (generative probabilities only)...{Style.RESET}")
    
    start_time = time.time()
    engine.game = game  # Set the game for the engine
    best_move = engine.play()
    end_time = time.time()
    
    print(f"{Style.GREEN}Best move with shallow search: {best_move.uci()}{Style.RESET}")
    print(f"{Style.BLUE}Time taken: {end_time - start_time:.2f} seconds{Style.RESET}")
    
    # Apply the move
    board.push(best_move)
    game = Game(board)
    display_board(board)
    
    # Now try with tree search
    engine.shallow = False
    print(f"\n{Style.BLUE}Using tree search...{Style.RESET}")
    
    start_time = time.time()
    engine.game = game  # Update the game for the engine
    best_move = engine.play()
    end_time = time.time()
    
    print(f"{Style.GREEN}Best move with tree search: {best_move.uci()}{Style.RESET}")
    print(f"{Style.BLUE}Time taken: {end_time - start_time:.2f} seconds{Style.RESET}")
    
    # Apply the move
    board.push(best_move)
    display_board(board)
    
    # Demonstrate position evaluation
    engine.game = Game(board)
    eval_score = engine.evaluate_position()
    print(f"\n{Style.BLUE}Position evaluation: {eval_score:.4f}{Style.RESET}")
    print(f"{Style.BLUE}Positive values favor White, negative values favor Black{Style.RESET}")
    
    print(f"\n{Style.GREEN}Note: In a real application, you would use a pre-trained model.{Style.RESET}")
    print(f"{Style.GREEN}The untrained model is used here for demonstration purposes only.{Style.RESET}")

if __name__ == "__main__":
    main() 