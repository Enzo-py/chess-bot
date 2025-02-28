import chess
from models.cnn.cnn_score import CNNScore
from models.deep_engine import *

from src.chess.game_generator import GameGenerator
from src.chess.game import Game
from src.chess.puzzle import Puzzle
from src.chess.loader import Loader

from src.utils.console import Style

import tqdm
import torch
import random

if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Setup environment...")
    model = CNNScore()
    # model.load("backend/models/saves/CNNScore.active.pth")
    
    nb_batches = 4
    batch_size = 2
    epochs = 2
    test_size = 10

    games: list[Game] = []
    ld_games = Loader(window=nb_batches, epochs_per_window=10).load(
        "backend/data/lichess_db_standard_rated_2013-04.pgn.zst", 
        Game, 
        chunksize=batch_size
    )

    ld_puzzles = Loader(window=nb_batches, epochs_per_window=10).load(
        "backend/data/lichess_db_puzzle.csv.zst", 
        Puzzle, 
        chunksize=batch_size
    )

    print("Environment ready.")

                 
    with model | board_evaluation_head | with_prints | auto_save as env:
        env.train(
            epochs=epochs, 
            batch_size=batch_size, 
            loader=ld_games | ld_puzzles
        )

        env.test(loader=ld_games | ld_puzzles)

    