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
    
    nb_batches = 1
    batch_size = 2
    epochs = 8
    test_size = 10

    games: list[Game] = []
    ld = Loader()
    ld_puzzle = Loader()

    games_getter = ld.load("backend/data/lichess_db_standard_rated_2013-04.pgn.zst", Game, chunksize=batch_size)
    puzzles_getter = ld_puzzle.load("backend/data/lichess_db_puzzle.csv.zst", Puzzle, chunksize=batch_size)

    for _ in range(nb_batches*0 + test_size): next(games_getter) # Skip the first games
    print("Environment ready.")

    games_train = []
    moves_train = []
    # for _ in tqdm.tqdm(range(nb_batches)):
    #     for puzzle in next(puzzles_getter):
    #         if puzzle.fen is not None:
    #             for idx in range(len(puzzle.moves) - 1):
    #                 games_train.append(puzzle.game)
    #                 moves_train.append(puzzle.moves[0])
    #                 puzzle.game = puzzle.game.copy()
    #                 puzzle.game.move(puzzle.moves[0])
    #                 puzzle.moves = puzzle.moves[1:]

    for _ in tqdm.tqdm(range(nb_batches)):
        for game in next(games_getter):
            if game.fen is not None and len(game.history) > 2:
                for idx in range(len(game.history) - 2):
                    move = game.board.pop()
                    shallow_cpy = game.copy()
                    games_train.append(shallow_cpy)
                    moves_train.append(move)

    games_test = []
    moves_test = []

    # for _ in tqdm.tqdm(range(test_size)):
    #     for puzzle in next(puzzles_getter):
    #         if puzzle.fen is not None:
    #             games_test.append(puzzle.game)
    #             moves_test.append(puzzle.moves[0])

    for _ in tqdm.tqdm(range(test_size)):
        for game in next(games_getter):
            if game.fen is not None and len(game.history) > 8:
                for idx in range(random.randint(1, len(game.history) - 8)):
                    game.board.pop()
                
                move = game.board.pop()
                games_test.append(game)
                moves_test.append(move)
                        
    with model | generative_head | on_games | with_prints | auto_save as env:
        env.train(epochs=epochs, batch_size=batch_size, games=games_train, moves=moves_train)
        env.test(games=games_test, moves=moves_test)

    
    