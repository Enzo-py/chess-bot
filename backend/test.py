import chess
from models.deep_engine import DeepEngine

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
    model = DeepEngine()
    
    nb_batches = 20
    batch_size = 64
    epochs = 20
    test_size = 10

    games: list[Game] = []
    ld = Loader()
    ld_puzzle = Loader()

    games_getter = ld.load("backend/data/lichess_db_standard_rated_2013-04.pgn.zst", Game, chunksize=batch_size)
    puzzles_getter = ld_puzzle.load("backend/data/lichess_db_puzzle.csv.zst", Puzzle, chunksize=batch_size)

    for _ in range(nb_batches*2 + test_size): next(games_getter) # Skip the first games
    print("Environment ready.")

    puzzles = []
    moves = []
    for _ in tqdm.tqdm(range(nb_batches)):
        for puzzle in next(puzzles_getter):
            if puzzle.fen is not None:
                puzzles.append(puzzle.game)
                moves.append(puzzle.moves[0])


    model.train(puzzles, moves, 10)


    # 1. test
    acc = model.evaluate(puzzles, moves)
    print(acc)

    # 2. train the model on legal moves

    # 2.1 get some games
    # for _ in tqdm.tqdm(range(nb_batches)):
    #     games.extend(next(games_getter))

    # # 2.2 randomly rewind the games
    # for game in games:
    #     if len(game.history) <= 1: continue
    #     game.rewind(random.randint(0, len(game.history) - 1))

    # 2.3 get the data correctly formatted
    # one_hots = torch.stack([torch.tensor(game.one_hot(), dtype=torch.float) for game in games])
    # turns = torch.tensor([int(game.board.turn) for game in games], dtype=torch.float)
    # y = [[move.uci() for move in game.board.legal_moves] for game in games]

    # one_hots, turns = one_hots.to(device), turns.to(device)

    # # 2.4 train the model
    # model.fit_legal_moves(one_hots, turns, y, epochs=epochs)

    # model.save("backend/models/cnn/cnn_score_legal_moves.pth", "legal-move-classifier")
    # model.save("backend/models/cnn/cnn_score_embedding.pth", "feature-extractor")

    # # ----------------------------------------------------------------------------
    # # 2. test the legal moves head --> (initial) 0.21% accuracy
    # games = []
    # # 2.1 get some games
    # for _ in tqdm.tqdm(range(10)):
    #     games.extend(next(games_getter))

    # # 2.2 randomly rewind the games
    # for game in games:
    #     if len(game.history) <= 1: continue
    #     game.rewind(random.randint(0, len(game.history) - 1))

    # # 2.3 get the data correctly formatted
    # one_hots = [game.one_hot() for game in games]
    # turns = [int(game.board.turn) for game in games]
    # y = [[move.uci() for move in game.board.legal_moves] for game in games]

    # # 2.4 evaluate the model
    # acc = model.evaluate_legal_moves(one_hots, turns, y)
    # print(f"Accuracy before training: {acc:.2%}")

    # ----------------------------------------------------------------------------
    # 3. train the model on the game outcome

    # # 3.1 get some games
    # for _ in tqdm.tqdm(range(nb_batches)):
    #     games.extend(next(games_getter))

    # # 3.2 randomly rewind the games
    # for game in games:
    #     if len(game.history) <= 1: continue
    #     game.rewind(random.randint(min(len(game.history) - 2, 1), min(len(game.history) - 1, 4)))

    # # 3.3 get the data correctly formatted
    # one_hots = torch.stack([torch.tensor(game.one_hot(), dtype=torch.float) for game in games])
    # turns = torch.tensor([int(game.board.turn) for game in games], dtype=torch.float)
    # y = [int(game.winner) for game in games]

    # one_hots, turns = one_hots.to(device), turns.to(device)

    # # 3.4 train the model
    # model.fit(one_hots, turns, y, epochs=epochs)

    model.save("backend/models/cnn/cnn_score_embedding.pth", "feature-extractor")
    model.save("backend/models/cnn/cnn_score_classifier.pth", "score-classifier")


    # 1. test the model without training --> (initial) 12% accuracy
    # games = []
    # # 1.1 get some games
    # for _ in tqdm.tqdm(range(10)):
    #     games.extend(next(games_getter))

    # # 1.2 randomly rewind the games
    # for game in games:
    #     if len(game.history) <= 1: continue
    #     game.rewind(random.randint(min(len(game.history) - 2, 1), min(len(game.history) - 1, 4)))

    # # 1.3 get the data correctly formatted
    # one_hots = [game.one_hot() for game in games]
    # turns = [int(game.board.turn) for game in games]
    # y = [int(game.winner) for game in games]

    # # 1.4 evaluate the model
    # acc = model.evaluate(one_hots, turns, y)
    # print(f"Accuracy before training: {acc:.2%}")


    