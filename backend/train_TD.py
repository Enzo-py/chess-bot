import os
from models.rl.td import TDLearningAI
from models.rl.mcts import MonteCarloAI
from models.deep_engine import *
from src.chess.loader import Loader, Game

# Path to Lichess database file
pgn_zst_path = os.path.join("backend", "data", "lichess_db_standard_rated_2015-05.pgn.zst")
# Save the intermediate tables during training
save_intermediate_paths = False

if not os.path.exists(pgn_zst_path):
    print(f"Error: Database file not found at {pgn_zst_path}")
    print("Please download the Lichess database file and place it in the correct location")
    exit(1)

print(f"Starting TD learning using database: {pgn_zst_path}")


game = Game()
td_agent = TDLearningAI()
td_agent.game = game

# Chose training parameters
training_stages = [
    {"epsilon": 0.5, "num_games": 30000},
    {"epsilon": 0.3, "num_games": 50000},
    {"epsilon": 0.1, "num_games": 50000},
]

for i, stage in enumerate(training_stages):
    print(f"\nTraining stage {i+1}/{len(training_stages)}")
    print(f"Epsilon: {stage['epsilon']}, Games: {stage['num_games']}")
    td_agent.epsilon = stage['epsilon']
    td_agent.learn(num_games=stage['num_games'])
    
    if save_intermediate_paths:
        save_path = f"backend/models/saves/td_value_table_stage_{i+1}.json"
        td_agent.save_value_table(save_path)
        print(f"Intermediate value table saved to {save_path}")

# Save the final value table
td_agent.save_value_table("backend/models/saves/td_value_table_final.json")
print(f"\nTraining complete. Final value table saved to td_value_table_final.json")
print(f"Total positions learned: {len(td_agent.value_table)}")