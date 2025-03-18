import chess
import json
import tqdm
import asyncio
import torch
import os

from src.chess.game import Game
from meta import AVAILABLE_MODELS

# Initialize the Stockfish engine (adjust the path to your Stockfish binary if needed)
engine_path = "/opt/homebrew/bin/stockfish"  # assumes Stockfish is in PATH; otherwise provide full path
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Engine analysis parameters
ANALYSIS_DEPTH = 8  # Lower depth for quicker analysis
BLUNDER_THRESHOLD = 100  # centipawn threshold for counting a blunder (e.g., 100 cp = 1 pawn)

# Define the path for Stockfish evaluation results
stockfish_eval_path = "data/stockfish_eval_results.json"

# Ensure the Stockfish evaluation results file exists
if not os.path.exists(stockfish_eval_path):
    with open(stockfish_eval_path, "w") as f:
        json.dump([], f)

# Function to save evaluation results to a JSON file
def save_evaluation_results(results):
    with open(stockfish_eval_path, "r") as f:
        existing_results = json.load(f)
    existing_results.append(results)
    with open(stockfish_eval_path, "w") as f:
        json.dump(existing_results, f, indent=4)

def evaluate_game(ai_white_name, ai_black_name):
    """
    Play a game between ai_white and ai_black, using Stockfish to evaluate each move.
    Returns a dictionary with performance metrics for both AIs in this game.
    """
    ai_white = AVAILABLE_MODELS[ai_white_name]()
    ai_black = AVAILABLE_MODELS[ai_black_name]()

    # Setup the AI models
    ai_white.setup()
    ai_black.setup()

    # Initialize the game for each AI
    game = Game()
    ai_white.game = game
    ai_black.game = game

    # Set the board for the game
    game.board = chess.Board()

    # Metrics accumulators
    total_cp_loss_white = 0   # total centipawn loss for White
    total_cp_loss_black = 0   # total centipawn loss for Black
    blunders_white = 0
    blunders_black = 0
    moves_white = 0
    moves_black = 0

    # Play until game over
    while not game.board.is_game_over():
        # Determine whose turn it is
        if game.board.turn == chess.WHITE:
            ai_player = ai_white
            perspective = chess.WHITE  # evaluate from White's perspective
        else:
            ai_player = ai_black
            perspective = chess.BLACK  # evaluate from Black's perspective

        # Get Stockfish's evaluation for the current position (before the move)
        info_before = engine.analyse(game.board, limit=chess.engine.Limit(depth=ANALYSIS_DEPTH))
        score_before = info_before["score"].pov(perspective)  # engine score from the player's perspective
        # Convert score to centipawns (handle mate scores if any)
        if score_before.is_mate():
            # If a mate is found, assign a very high value (10000 for mate win, -10000 for mate loss)
            best_score_cp = 10000 if score_before.mate() > 0 else -10000
        else:
            best_score_cp = score_before.score()  # centipawn score

        # Ask the AI player to play a move
        move = ai_player.play()  # Use the play method
        game.board.push(move)  # make the move on the board

        # Get Stockfish's evaluation after the move
        info_after = engine.analyse(game.board, limit=chess.engine.Limit(depth=ANALYSIS_DEPTH))
        # Now it's the opponent's turn, but we still interpret the score from the perspective of the player who just moved
        score_after = info_after["score"].pov(perspective)
        if score_after.is_mate():
            # Handle mate scores similarly
            actual_score_cp = 10000 if score_after.mate() > 0 else -10000
        else:
            actual_score_cp = score_after.score()

        # Calculate centipawn loss for the move
        cp_loss = best_score_cp - actual_score_cp
        # (If cp_loss is negative, it means the move *improved* the position beyond the engine's expectation,
        # which can happen if multiple moves are equally good or the engine's depth was limited. 
        # We can treat negative loss as 0 for performance metrics, since it means no mistake was made.)
        if cp_loss < 0:
            cp_loss = 0

        # Accumulate metrics for the respective player
        if perspective == chess.WHITE:
            moves_white += 1
            total_cp_loss_white += cp_loss
            if cp_loss >= BLUNDER_THRESHOLD:
                blunders_white += 1
        else:
            moves_black += 1
            total_cp_loss_black += cp_loss
            if cp_loss >= BLUNDER_THRESHOLD:
                blunders_black += 1

    # Game is over, compute average metrics
    acpl_white = total_cp_loss_white / moves_white if moves_white > 0 else 0.0
    acpl_black = total_cp_loss_black / moves_black if moves_black > 0 else 0.0

    # You could also record the final result if needed (e.g., board.result())
    return {
        "acpl_white": acpl_white,
        "acpl_black": acpl_black,
        "blunders_white": blunders_white,
        "blunders_black": blunders_black
    }

# Example usage: simulate multiple games and aggregate performance

ais = ["Random AI", "Sunfish AI", "Greedy AI", "Transformer AI", "Tree Search Transformer", "Score CNN", "GreedyExploration AI", "MCTS AI", "Q-Learning AI", "Stockfish AI"]  # list of AI player names

# Number of games each AI plays against each other
N = 5  # Reduce the number of games to speed up execution

# Results dictionary to accumulate metrics
results = {ai: {"games": 0, "total_acpl": 0.0, "total_blunders": 0} for ai in ais}

# Play games between each pair of AIs
for ai_white in ais:
    for ai_black in ais:
        if ai_white == ai_black:
            continue
        # Play N games with ai_white vs ai_black
        for game_index in range(N):
            metrics = evaluate_game(ai_white, ai_black)
            # Save the evaluation results to the JSON file
            save_evaluation_results({"white": ai_white, "black": ai_black, "metrics": metrics})
            # Accumulate metrics for ai_white and ai_black
            results[ai_white]["games"] += 1
            results[ai_black]["games"] += 1
            results[ai_white]["total_acpl"] += metrics["acpl_white"]
            results[ai_black]["total_acpl"] += metrics["acpl_black"]
            results[ai_white]["total_blunders"] += metrics["blunders_white"]
            results[ai_black]["total_blunders"] += metrics["blunders_black"]

# Compute average performance metrics for each AI across all games
for ai in ais:
    if results[ai]["games"] > 0:
        avg_acpl = results[ai]["total_acpl"] / results[ai]["games"]
        avg_blunders = results[ai]["total_blunders"] / results[ai]["games"]
        print(f"{ai}: Average ACPL = {avg_acpl:.2f}, Average Blunders per game = {avg_blunders:.2f}")
    else:
        print(f"{ai}: No games played.")

# Close the engine after analysis is done
engine.close()
