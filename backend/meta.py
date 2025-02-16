from models.random_ai import RandomAI
from models.greedy_ai import GreedyAI
from models.greedy_exploration import GreedyExplorationAI
from models.stockfish import StockfishAI
from models.alpha_beta import SunfishAI
from models.mcts import MonteCarloAI

AVAILABLE_MODELS = {
    "Random AI": RandomAI,
    "Greedy AI": GreedyAI,
    "GreedyExploration AI": GreedyExplorationAI,
    "Stockfish AI": StockfishAI,
    "Sunfish AI": SunfishAI,
    "MCTS AI": MonteCarloAI,
}
"""
This dictionary exposes all the models that are available for testing or using in the interface.
The key is the model name and the value is the model class.

WARNING: Do not change the keys of this dictionary. Use in backend/data/ranking.json
"""

CHESS_PROBLEM = [
    {
        "fen": '3rr3/1b4kp/nbn1Qpp1/pp1N4/P4BPq/1P1B1P1P/2P5/2K1NR1R w - - 4 30',
        "solution": ['Qxe8', 'Rxe8', 'Ng2', 'Fe3+', 'Cdxe3', 'Rxe3', 'Cxh4'],
        "description": "White to move, take the black queen",
    },
]
