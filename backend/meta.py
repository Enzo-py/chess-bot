from models.greedy.random_ai import RandomAI
from models.greedy.greedy_ai import GreedyAI
from models.greedy.greedy_exploration import GreedyExplorationAI
from models.downloaded.stockfish import StockfishAI
from models.rl.alpha_beta import AlphaBetaSearchAI
from models.rl.mcts import MonteCarloAI
from models.rl.q_learning import QLearningAI
from models.cnn.cnn_score import CNNScore
from models.cnn.tree_search_cnn import TreeSearchCNN
from models.rl.td import TDLearningAI
from models.cnn2.cnn_score2 import CNNScore2
from models.transformer.transformer import TransformerScore
from models.transformer.alpha_beta import AlphaBetaTransformerEngine
from models.transformer.tree_search import TreeSearchTransformer

AVAILABLE_MODELS = {
    "Random AI": RandomAI,
    "Greedy AI": GreedyAI,
    "GreedyExploration AI": GreedyExplorationAI,
    "Stockfish AI": StockfishAI,
    "AlphaBetaSearchAI": AlphaBetaSearchAI,
    "MCTS AI": MonteCarloAI,
    "Q-Learning AI": QLearningAI,
    "Score CNN": CNNScore,
    # "Tree Search CNN": TreeSearchCNN,
    "Score CNN2": CNNScore2,
    "TD Learning AI": TDLearningAI,
    "Transformer AI": TransformerScore,
    "Alpha-Beta Transformer": AlphaBetaTransformerEngine,
    "Tree Search Transformer": TreeSearchTransformer,
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
