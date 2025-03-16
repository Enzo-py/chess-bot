import sys
import os
import time
import chess

# Add parent directory to path to import from models
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from models.transformer.alpha_beta import AlphaBetaTransformerEngine, AlphaBetaTransformerEngineWithSorting
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
    
    # Create an instance of the AlphaBetaTransformerEngine
    engine = AlphaBetaTransformerEngine(max_depth=3)
    print(f"{Style.GREEN}Created AlphaBetaTransformerEngine with depth 3{Style.RESET}")
    
    # Get the best move using the engine
    print(f"{Style.BLUE}Calculating best move...{Style.RESET}")
    start_time = time.time()
    best_move = engine.choose_move(game)
    end_time = time.time()
    print(f"{Style.GREEN}Best move according to transformer alpha-beta: {best_move.uci()}{Style.RESET}")
    print(f"{Style.BLUE}Time taken: {end_time - start_time:.2f} seconds{Style.RESET}")
    
    # Apply the move
    board.push(best_move)
    game = Game(board)
    display_board(board)
    
    # Try with move ordering for comparison
    enhanced_engine = AlphaBetaTransformerEngineWithSorting(max_depth=3)
    print(f"{Style.GREEN}Created AlphaBetaTransformerEngineWithSorting with depth 3{Style.RESET}")
    
    # Get the best move using the enhanced engine
    print(f"{Style.BLUE}Calculating best move with move ordering...{Style.RESET}")
    start_time = time.time()
    best_move = enhanced_engine.choose_move(game)
    end_time = time.time()
    print(f"{Style.GREEN}Best move with move ordering: {best_move.uci()}{Style.RESET}")
    print(f"{Style.BLUE}Time taken: {end_time - start_time:.2f} seconds{Style.RESET}")
    
    # Apply the move
    board.push(best_move)
    display_board(board)
    
    # Example of position evaluation
    eval_score = enhanced_engine.evaluate_board(Game(board))
    print(f"{Style.BLUE}Position evaluation: {eval_score:.4f}{Style.RESET}")
    print(f"{Style.BLUE}Positive values favor White, negative values favor Black{Style.RESET}")
    
if __name__ == "__main__":
    main() 