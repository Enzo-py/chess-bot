import chess
from models.cnn.cnn_score import CNNScore
from models.deep_engine import *

from src.chess.game_generator import GameGenerator
from src.chess.game import Game
from src.chess.puzzle import Puzzle
from src.chess.loader import Loader

if __name__ == "__main__":
    
    print("Setup environment...")
    model = CNNScore()
    model.load("backend/models/saves/CNNScore-V20.auto-save.pth")
    
    nb_batches = 5
    batch_size = 32
    epochs = 10
    test_size = 18

    games: list[Game] = []
    ld_games = Loader(window=nb_batches, epochs_per_window=3, min_elo=2000).load(
        "backend/data/lichess_db_standard_rated_2013-04.pgn.zst", 
        Game, 
        chunksize=batch_size
    )

    ld_puzzles = Loader(window=nb_batches, epochs_per_window=3).load(
        "backend/data/lichess_db_puzzle.csv.zst", 
        Puzzle, 
        chunksize=batch_size
    )

    # ld_puzzles.skip(nb_batches*epochs*2)
    # ld_games.skip(nb_batches*epochs*2)

    print("Environment ready.")

    # with model | encoder_head | with_prints | auto_save as env:
    #     env.train(
    #         epochs=epochs, 
    #         batch_size=batch_size, 
    #         loader=ld_games | ld_puzzles
    #     )

    #     env.test(loader=ld_games | ld_puzzles)

    # with model | generative_head | with_prints | auto_save as env:
    #     env.train(
    #         epochs=epochs, 
    #         batch_size=batch_size, 
    #         loader=ld_games
    #     )

    #     env.test(loader=ld_games | ld_puzzles)
       
    with model | board_evaluation_head | with_prints | auto_save as env:
        # env.train(
        #     epochs=epochs, 
        #     batch_size=batch_size, 
        #     loader=ld_games | ld_puzzles
        # )

        env.test(loader=ld_games | ld_puzzles)

    # with model | generative_head | with_prints | auto_save as env:
    #     env.train(
    #         epochs=epochs, 
    #         batch_size=batch_size, 
    #         loader=ld_games
    #     )

    #     env.test(loader=ld_games | ld_puzzles)
       
    with model | board_evaluation_head | with_prints | auto_save as env:
        # env.train(
        #     epochs=epochs, 
        #     batch_size=batch_size, 
        #     loader=ld_games | ld_puzzles
        # )

        env.test(loader=ld_games | ld_puzzles)

    