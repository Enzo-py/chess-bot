import chess
import json
import tqdm
import asyncio
import torch
import os
import numpy as np

from src.chess.game import Game
from meta import AVAILABLE_MODELS

# Initialize the Stockfish engine (adjust the path to your Stockfish binary if needed)
engine_path = "/opt/homebrew/bin/stockfish"  # assumes Stockfish is in PATH; otherwise provide full path
engine = chess.engine.SimpleEngine.popen_uci(engine_path)

# Engine analysis parameters
ANALYSIS_DEPTH = 8  # Lower depth for quicker analysis
MOVE_LIMIT = 100   # Maximum number of moves per game
MOVE_QUALITY_THRESHOLDS = {
    'brilliant': 5,    # <= 5 cp loss
    'excellent': 15,   # <= 15 cp loss
    'good': 30,        # <= 30 cp loss
    'inaccuracy': 80,  # <= 80 cp loss
    'mistake': 150,    # <= 150 cp loss
    'blunder': float('inf')  # > 150 cp loss
}

# Define the path for Stockfish evaluation results
stockfish_eval_path = "data/results_stockfish.json"

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

def get_game_phase(board):
    """
    Determine the phase of the game based on piece count and position
    Returns: 'opening', 'middlegame', or 'endgame'
    """
    # Count material
    total_pieces = len(board.piece_map())
    
    if total_pieces >= 28:  # Most pieces still on board
        return 'opening'
    elif total_pieces <= 12:  # Few pieces left
        return 'endgame'
    else:
        return 'middlegame'

def classify_move_quality(cp_loss):
    """Classify the quality of a move based on centipawn loss"""
    for quality, threshold in MOVE_QUALITY_THRESHOLDS.items():
        if cp_loss <= threshold:
            return quality
    return 'blunder'

def evaluate_game(ai_white_name, ai_black_name):
    """
    Play a game between ai_white and ai_black, using Stockfish to evaluate each move.
    Returns a dictionary with comprehensive performance metrics for both AIs.
    Game will terminate after MOVE_LIMIT moves if no conclusion is reached.
    """
    ai_white = AVAILABLE_MODELS[ai_white_name]()
    ai_black = AVAILABLE_MODELS[ai_black_name]()

    # Setup the AI models
    ai_white.setup()
    ai_black.setup()

    # Initialize the game
    game = Game()
    ai_white.game = game
    ai_black.game = game
    game.board = chess.Board()

    # Metrics accumulators for each player
    metrics = {
        'white': {
            'moves_count': 0,
            'total_cp_loss': 0,
            'position_scores': [],
            'move_qualities': {quality: 0 for quality in MOVE_QUALITY_THRESHOLDS.keys()},
            'phase_performance': {'opening': [], 'middlegame': [], 'endgame': []},
            'mate_sequences_found': 0,
            'mate_sequences_missed': 0,
            'avg_position_complexity': 0,
            'decisive_moves': 0,
        },
        'black': {
            'moves_count': 0,
            'total_cp_loss': 0,
            'position_scores': [],
            'move_qualities': {quality: 0 for quality in MOVE_QUALITY_THRESHOLDS.keys()},
            'phase_performance': {'opening': [], 'middlegame': [], 'endgame': []},
            'mate_sequences_found': 0,
            'mate_sequences_missed': 0,
            'avg_position_complexity': 0,
            'decisive_moves': 0,
        }
    }

    total_moves = 0
    # Play until game over or move limit reached
    while not game.board.is_game_over() and total_moves < MOVE_LIMIT:
        total_moves += 1
        current_phase = get_game_phase(game.board)
        
        # Determine whose turn it is
        if game.board.turn == chess.WHITE:
            ai_player = ai_white
            player_metrics = metrics['white']
            perspective = chess.WHITE
        else:
            ai_player = ai_black
            player_metrics = metrics['black']
            perspective = chess.BLACK

        # Get Stockfish's evaluation before the move
        info_before = engine.analyse(game.board, limit=chess.engine.Limit(depth=ANALYSIS_DEPTH))
        score_before = info_before["score"].pov(perspective)
        
        # Convert score to centipawns
        if score_before.is_mate():
            best_score_cp = 10000 if score_before.mate() > 0 else -10000
        else:
            best_score_cp = score_before.score()

        # Store position score
        player_metrics['position_scores'].append(best_score_cp)

        # Play the move
        move = ai_player.play()
        if move is None:  # Handle case where AI can't make a move
            break
        game.board.push(move)

        # Get Stockfish's evaluation after the move
        info_after = engine.analyse(game.board, limit=chess.engine.Limit(depth=ANALYSIS_DEPTH))
        score_after = info_after["score"].pov(perspective)
        
        if score_after.is_mate():
            actual_score_cp = 10000 if score_after.mate() > 0 else -10000
        else:
            actual_score_cp = score_after.score()

        # Calculate centipawn loss
        cp_loss = max(0, best_score_cp - actual_score_cp)

        # Update metrics
        player_metrics['moves_count'] += 1
        player_metrics['total_cp_loss'] += cp_loss
        
        # Classify move quality
        move_quality = classify_move_quality(cp_loss)
        player_metrics['move_qualities'][move_quality] += 1
        
        # Track performance by game phase
        player_metrics['phase_performance'][current_phase].append(cp_loss)

        # Check for mate sequences
        if score_before.is_mate() and not score_after.is_mate():
            player_metrics['mate_sequences_missed'] += 1
        elif not score_before.is_mate() and score_after.is_mate() and score_after.mate() > 0:
            player_metrics['mate_sequences_found'] += 1

        # Track decisive moves (evaluation change > 200 cp)
        if abs(actual_score_cp - best_score_cp) > 200:
            player_metrics['decisive_moves'] += 1

    # Add game termination reason
    if total_moves >= MOVE_LIMIT:
        metrics['termination_reason'] = 'move_limit'
    elif game.board.is_checkmate():
        metrics['termination_reason'] = 'checkmate'
    elif game.board.is_stalemate():
        metrics['termination_reason'] = 'stalemate'
    elif game.board.is_insufficient_material():
        metrics['termination_reason'] = 'insufficient_material'
    elif game.board.is_fifty_moves():
        metrics['termination_reason'] = 'fifty_moves'
    elif game.board.is_repetition():
        metrics['termination_reason'] = 'repetition'
    else:
        metrics['termination_reason'] = 'other'

    # Calculate final metrics
    for color in ['white', 'black']:
        m = metrics[color]
        moves = m['moves_count']
        if moves > 0:
            m['acpl'] = m['total_cp_loss'] / moves
            m['move_quality_percentages'] = {
                quality: (count / moves) * 100 
                for quality, count in m['move_qualities'].items()
            }
            
            for phase in ['opening', 'middlegame', 'endgame']:
                phase_moves = m['phase_performance'][phase]
                m['phase_performance'][phase] = np.mean(phase_moves) if phase_moves else 0
            
            if len(m['position_scores']) > 1:
                m['position_volatility'] = np.std(m['position_scores'])
            else:
                m['position_volatility'] = 0
            
            m['decisive_move_percentage'] = (m['decisive_moves'] / moves) * 100

    m['total_moves'] = total_moves
    return metrics

# Example usage: simulate multiple games and aggregate performance
ais = ["Random AI", "Sunfish AI", "Greedy AI", "Transformer AI", "Tree Search Transformer", 
       "Score CNN", "GreedyExploration AI", "MCTS AI", "Q-Learning AI", "Stockfish AI"]

# Number of games each AI plays against each other
N = 5  # Reduced number of games for faster execution

# Results dictionary to accumulate metrics
results = {ai: {
    'games_played': 0,
    'total_moves': 0,
    'total_acpl': 0,
    'move_qualities': {quality: 0 for quality in MOVE_QUALITY_THRESHOLDS.keys()},
    'phase_performance': {'opening': [], 'middlegame': [], 'endgame': []},
    'mate_sequences': {'found': 0, 'missed': 0},
    'position_volatility': [],
    'decisive_moves': 0
} for ai in ais}

# Play games between each pair of AIs
for ai_white in ais:
    for ai_black in ais:
        if ai_white == ai_black:
            continue
        
        for _ in range(N):
            metrics = evaluate_game(ai_white, ai_black)
            save_evaluation_results({
                "white": ai_white,
                "black": ai_black,
                "metrics": metrics
            })
            
            # Update white's metrics
            results[ai_white]['games_played'] += 1
            results[ai_white]['total_moves'] += metrics['white']['moves_count']
            results[ai_white]['total_acpl'] += metrics['white']['total_cp_loss']
            for quality in MOVE_QUALITY_THRESHOLDS.keys():
                results[ai_white]['move_qualities'][quality] += metrics['white']['move_qualities'][quality]
            results[ai_white]['mate_sequences']['found'] += metrics['white']['mate_sequences_found']
            results[ai_white]['mate_sequences']['missed'] += metrics['white']['mate_sequences_missed']
            results[ai_white]['position_volatility'].append(metrics['white']['position_volatility'])
            results[ai_white]['decisive_moves'] += metrics['white']['decisive_moves']
            
            # Update black's metrics similarly
            results[ai_black]['games_played'] += 1
            results[ai_black]['total_moves'] += metrics['black']['moves_count']
            results[ai_black]['total_acpl'] += metrics['black']['total_cp_loss']
            for quality in MOVE_QUALITY_THRESHOLDS.keys():
                results[ai_black]['move_qualities'][quality] += metrics['black']['move_qualities'][quality]
            results[ai_black]['mate_sequences']['found'] += metrics['black']['mate_sequences_found']
            results[ai_black]['mate_sequences']['missed'] += metrics['black']['mate_sequences_missed']
            results[ai_black]['position_volatility'].append(metrics['black']['position_volatility'])
            results[ai_black]['decisive_moves'] += metrics['black']['decisive_moves']

# Print comprehensive performance analysis for each AI
print("\nComprehensive AI Performance Analysis")
print("=" * 50)

for ai in ais:
    if results[ai]['games_played'] > 0:
        r = results[ai]
        moves = r['total_moves']
        print(f"\n{ai}:")
        print(f"Games played: {r['games_played']}")
        print(f"Average moves per game: {moves / r['games_played']:.2f}")
        print(f"Average centipawn loss: {r['total_acpl'] / moves:.2f}")
        
        print("\nMove Quality Distribution:")
        for quality in MOVE_QUALITY_THRESHOLDS.keys():
            percentage = (r['move_qualities'][quality] / moves) * 100
            print(f"  {quality}: {percentage:.1f}%")
        
        print("\nMate Sequences:")
        print(f"  Found: {r['mate_sequences']['found']}")
        print(f"  Missed: {r['mate_sequences']['missed']}")
        
        print(f"\nPosition Volatility: {np.mean(r['position_volatility']):.2f}")
        print(f"Decisive Moves per Game: {r['decisive_moves'] / r['games_played']:.2f}")
    else:
        print(f"\n{ai}: No games played.")

# Close the engine after analysis is done
engine.close()
