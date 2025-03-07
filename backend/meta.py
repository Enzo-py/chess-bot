from models.greedy.random_ai import RandomAI
from models.greedy.greedy_ai import GreedyAI
from models.greedy.greedy_exploration import GreedyExplorationAI
from models.downloaded.stockfish import StockfishAI
from models.rl.alpha_beta import SunfishAI
from models.rl.mcts import MonteCarloAI
from models.cnn.cnn_score import CNNScore
from models.rl.td import TDLearningAI
from models.cnn2.cnn_score2 import CNNScore2

AVAILABLE_MODELS = {
    "Random AI": RandomAI,
    "Greedy AI": GreedyAI,
    "GreedyExploration AI": GreedyExplorationAI,
    "Stockfish AI": StockfishAI,
    "Sunfish AI": SunfishAI,
    "MCTS AI": MonteCarloAI,
    "Score CNN": CNNScore,
    "Score CNN2": CNNScore2,
    "TD Learning AI": TDLearningAI,
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
