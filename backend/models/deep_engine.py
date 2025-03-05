from functools import wraps
import random
from typing import Union

from src.chess.puzzle import Puzzle
from src.chess.loader import Loader
from src.utils.console import Style, deprecated
from src.chess.game import Game
from models.engine import Engine
from models.train_config import *
from models.template import DefaultClassifier, DefaultDecoder, Embedding, DefaultGenerativeHead



import chess
import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeepEngine", "all_heads", "generative_head", "board_evaluation_head", "encoder_head", "with_prints", "auto_save"]

class DeepEngine(Engine):
    """
    Extention of the Engine class to be able to manage deep learning models.
    This class is abstract and should be inherited by a specific model.

    The DeepEngine will auto load the model from the `./models/saves/class_name.active.pth` file if it exists.
    As in engine, you have to implement the `play` method to play a move. By default, it will play the move predict by the generative head.
    
    DeepEngine declaration example:
    ------------------------------
    ```python
    class MyModel(DeepEngine):
        def __init__(self):
            super().__init__()
            # Board -- Encoder --> Latent
            self.set(head_name="encoder", nn.Linear(8*8*13, 64))

            # Latent -- Classifier --> Win probability (black, white)
            self.set(head_name="board_evaluation", nn.Linear(64, 2))

            # Latent -- Generative --> Move
            self.set(head_name="generative", nn.Linear(64, 64*64*5))

            # Latent -- Decoder --> Board
            self.set(head_name="decoder", nn.Linear(64, 8*8*12).reshape(-1, 8, 8, 12))
    ```

    For more information about the heads, see the documentation of the `DeepEngineModule` class.
    
    Trainning the model:
    --------------------

    ```python
    model = MyModel()
    model.load("path/to/model.pth")

    loader = Loader(...) # See loader documentation

    print(model.manifest) # Get the model manifest

    # Train the model
    with model | {head_name} | {UI} | {auto_save} as env:
        env.train(epochs=10, batch_size=32, loader=loader)
        env.test(loader=loader)

    ```

    UI can be `with_prints` or nothing.
    Auto save can be `auto_save` or nothing.

    Information
    -----------

    DeepEngine allow the developper to create a model easly in the chess / engine context.
    The train functions laverage different solution to make the trainning more efficient (such as contrastive loss, ...)
    """

    __author__ = "Enzo Pinchon"
    __description__ = "Deep Engine"

    PROMOTION_TABLE = {
        None: 0,
        chess.QUEEN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.module = DeepEngineModule()
        """torch.nn.Module: The model used to score the games."""

        self.is_setup = False
    
        self.auto_save_version = None
        self._train_config = {"mode": None, "head": None, "UI": None, "auto_save": False, "load_policy": "all"}

    def __or__(self, other: TrainConfigBase) -> TrainConfig:
        """Permet d'enchaîner les configurations avec `|` et retourne le bon type."""
        return other.apply(self)
    
    def setup(self, *args, **kwargs):
        """
        Setup the model.
        """
        path = "./models/saves/" + self.__class__.__name__ + ".active.pth"
        if os.path.exists(path):
            self.load(path)
            self.is_setup = True
        else:
            raise FileNotFoundError(f"Model not found at {path}")
        
        return self
    
    @property
    def manifest(self):
        return {
            "author": self.__author__,
            "description": self.__description__,
            "heads": {
                "board_evaluation": type(self.module.classifier).__name__,
                "generative": type(self.module.generative_head).__name__,
                "encoder": type(self.module.embedding).__name__,
                "decoder": type(self.module.generative_head).__name__,
            },
            "settings": {}
        }
    
    def evaluate(self, game: Game) -> float:
        """
        Evaluate the game: return the win probability of the game (black wr, white, wr).
        """
        self.setup()
        self.game = game
        return self.predict(head="board_evaluation")

    def set(self, head_name: str, head: nn.Module):
        """
        Set a head in the model.
        Possibles heads are:
        - board_evaluation
        - generative
        - encoder
        - decoder
        """

        assert head_name in ["board_evaluation", "generative", "encoder", "decoder"], "Invalid head name."
        if head_name == "board_evaluation":
            self.module.classifier = head
        elif head_name == "generative":
            self.module.generative_head = head
        elif head_name == "encoder":
            self.module.embedding = head
        elif head_name == "decoder":
            self.module.decoder = head

        self.module.to(self.module.device)

    def play(self, **kwargs):
        # check if setup, not mandatory but avoid hard to debug errors
        if not self.is_setup: raise ValueError("Model not setup.")

        scores = self.predict()
        legal_moves = list(self.game.board.legal_moves)
        scores = [scores[self.encode_move(move, as_int=True)] for move in legal_moves]
        return legal_moves[scores.index(max(scores))]

    def encode_move(self, move: chess.Move, as_int=False) -> Union[int, tuple]:
        """
        Encode a move to a tuple.
        """

        if as_int:
            move_tuple = (move.from_square, move.to_square, DeepEngine.PROMOTION_TABLE[move.promotion])
            moves_dict = None
            if os.path.exists("backend/data/moves_dict.pth"):
                moves_dict = torch.load("backend/data/moves_dict.pth", weights_only=False)
            elif os.path.exists("data/moves_dict.pth"):
                moves_dict = torch.load("data/moves_dict.pth", weights_only=False)
            else:
                raise FileNotFoundError("Moves dict not found.")
            return moves_dict[move_tuple]

        return move.from_square, move.to_square, DeepEngine.PROMOTION_TABLE[move.promotion]

    def decode_move(self, move: Union[int, tuple]) -> chess.Move:
        """
        Decode a move from an integer or a tuple to a chess.Move object.
        """
        if isinstance(move, tuple): # (from, to, promotion)
            reverse_promotion_table = {v: k for k, v in DeepEngine.PROMOTION_TABLE.items()}
            return chess.Move(move[0], move[1], reverse_promotion_table[move[2]])
        
        # get move from dict
        moves_dict = None
        if os.path.exists("backend/data/moves_dict.pth"):
            moves_dict = torch.load("backend/data/moves_dict.pth", weights_only=False)
        elif os.path.exists("data/moves_dict.pth"):
            moves_dict = torch.load("data/moves_dict.pth", weights_only=False)
        else:
            raise FileNotFoundError("Moves dict not found.")
        
        return self.decode_move(tuple(list(moves_dict.keys())[move]))
    
    def load(self, path: str=None, element: str="all"):
        """
        Load the model from a file.
        """

        assert element in ["all", "embedding", "classifier"], "Invalid element to load."
        if path is None:
            path = "./models/saves/" + self.__class__.__name__ + ".active.pth"
            if not os.path.exists(path):
                path = "backend/models/saves/" + self.__class__.__name__ + ".active.pth"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Model not found at {path}")

        if element == "all":
            self.module.load_state_dict(torch.load(path, weights_only=True, map_location=self.module.device), strict=False)
        elif element == "embedding":
            self.module.embedding.load_state_dict(torch.load(path, weights_only=True, map_location=self.module.device), strict=False)
        elif element == "classifier":
            self.module.classifier.load_state_dict(torch.load(path, weights_only=True, map_location=self.module.device), strict=False)

    def save(self, path: str=None, element: str="all"):
        """
        Save the model to a file.
        """

        assert element in ["all", "embedding", "classifier"], "Invalid element to save."

        if path is None:
            try:
                nb_other_files = len([f for f in os.listdir("./models/saves") if self.__class__.__name__ in f])
                path = f"./models/saves/{self.__class__.__name__}-V{nb_other_files}.pth"
            except:
                nb_other_files = len([f for f in os.listdir("backend/models/saves") if self.__class__.__name__ in f])
                path = f"backend/models/saves/{self.__class__.__name__}-V{nb_other_files}.pth"

        if element == "all":
            torch.save(self.module.state_dict(), path)
        elif element == "embedding":
            torch.save(self.module.embedding.state_dict(), path)
        elif element == "classifier":
            torch.save(self.module.classifier.state_dict(), path)

    def _exctract_data(self, loader: Loader, epoch: int, _for="train") -> tuple:

        assert _for in ["train", "test"], "Invalid value for '_for'. Must be 'train' or 'test'."
        games, moves = [], []

        if self._train_config["UI"] == "prints":
            l = " > Updating data"
            print(f"| {l: <{TrainConfig.line_length-2}} |")

        for d in tqdm.tqdm(loader.get_update(epoch), ncols=TrainConfig.line_length+2, desc="| "):
            if isinstance(d, Game):
                if len(d.history) <= 3: continue

                if self._train_config["head"] == "board_evaluation":
                    probs = (0.5, 0.5) if d.draw else (1, 0) if d.winner == chess.WHITE else (0, 1)
                    for _ in range(len(d.history) - 2):
                        d = d.copy()
                        move = d.board.pop()
                        games.append(d)
                        moves.append(probs)
                    
                    continue

                if _for == "train":
                    for _ in range(len(d.history) - 2):
                        d = d.copy()
                        move = d.board.pop()
                        games.append(d)
                        moves.append(move)
                else:
                    d = d.copy()
                    for _ in range(random.randint(1, len(d.history) - 2)):
                        d.board.pop()
                    
                    move = d.board.pop()
                    games.append(d)
                    moves.append(move)

            elif isinstance(d, Puzzle):
                if len(d.moves) <= 1: continue

                if self._train_config["head"] == "board_evaluation":
                    probs = (0.5, 0.5) if d.game.draw else (1, 0) if d.game.winner == chess.WHITE else (0, 1)
                    for _ in range(len(d.moves) - 1):
                        if _for == "train":
                            games.append(d.game)
                            moves.append(probs)
                            d.game = d.game.copy()
                            d.game.move(d.moves[0])
                            d.moves = d.moves[1:]
                        else:
                            games.append(d.game)
                            moves.append(probs)

                    continue

                for _ in range(len(d.moves) - 1):
                    if _for == "train":
                        games.append(d.game)
                        moves.append(d.moves[0])
                        d.game = d.game.copy()
                        d.game.move(d.moves[0])
                        d.moves = d.moves[1:]
                    else:
                        games.append(d.game)
                        moves.append(d.moves[0])

            else:
                raise ValueError("Invalid data type in loader.")
            
        if self._train_config["UI"] == "prints":
            print("|" + " " * TrainConfig.line_length + "|")
            
        return games, moves

    def _train_board_evaluation(self, epochs: int, batch_size, games: list[Game], win_probs: list[tuple[float, float]], loader: Loader = None):
        """
        Train the model on the board evaluation head: predict the win probability.
        [black_win, white_win]

        :param epochs: number of epochs
        :type epochs: int

        :param batch_size: size of the batch
        :type batch_size: int

        :param games: list of games
        :type games: list[Game]

        :param win_probs: list of tuples (black_win_prob, white_win_prob)
        :type win_probs: list[tuple[float, float]]

        :param loader: loader to use to get the games
        :type loader: Loader

        :return: total_loss, contrastive_loss, classification_loss
        :rtype: tuple[list[float], list[float], list[float], list[float]]
        """

        if loader is None:
            assert len(games) == len(win_probs) > 0, "You need at least one game to train, and one label per game."
            assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
            assert isinstance(win_probs[0], tuple), "'win_probs' should be a list of tuples."

        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.0005)

        num_samples = len(games or [])
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ensure full coverage

        all_total_loss = []
        all_contrastive_loss = []
        all_classification_loss = []

        for epoch in range(epochs):
            total_loss = 0
            contrastive_loss_total = 0
            classification_loss_total = 0

            if loader and loader.need_update(epoch):
                games, win_probs = self._exctract_data(loader, epoch)

                num_samples = len(games)
                num_batches = (num_samples + batch_size - 1) // batch_size

            indices = torch.randperm(num_samples)

            for batch_idx in tqdm.tqdm(range(num_batches), ncols=TrainConfig.line_length+2, desc="| "):
                
                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_games = [games[i] for i in batch_indices]
                batch_win_probs = torch.tensor([win_probs[i] for i in batch_indices], dtype=torch.float32, device=self.module.device)

                batch_games_t = [game.reverse() for game in batch_games]

                # Predict the probabilities
                probs, embeddings = self.module(batch_games, head="board_evaluation")
                probs_t, embeddings_t = self.module(batch_games_t, head="board_evaluation")

                # Compute classification loss (Binary Cross Entropy)
                classification_loss = F.binary_cross_entropy(probs, batch_win_probs)

                # Compute contrastive loss (make embeddings for original and flipped boards similar)
                contrastive_loss = self._contrastive_loss(probs, probs_t)

                # Total loss
                loss = classification_loss + 0.15 * contrastive_loss

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                contrastive_loss_total += contrastive_loss.item()
                classification_loss_total += classification_loss.item()

            all_total_loss.append(total_loss / num_batches)
            all_contrastive_loss.append(contrastive_loss_total / num_batches)
            all_classification_loss.append(classification_loss_total / num_batches)

            if self._train_config["UI"] == 'prints':
                l1 = f"  [{epoch + 1}/{epochs}] Loss: {total_loss / num_batches:.2f}, Contrastive Loss: {contrastive_loss_total / num_batches:.2f}, " 
                l2 = f"  Classification Loss: {classification_loss_total / num_batches:.2f}"
                l1 = Style("INFO", f"{l1: <{TrainConfig.line_length-2}}")
                l2 = Style("INFO", f"{l2: <{TrainConfig.line_length-2}}")
                print(f"| {l1} |")
                print(f"| {l2} |")
                print(f"| {'': <{TrainConfig.line_length-2}} |")

            if self._train_config["auto_save"]:
                if self.auto_save_version is None:
                    files = [f for f in os.listdir("backend/models/saves") if self.__class__.__name__ in f and "auto-save" in f]
                    self.auto_save_version = len(files)
                self.save(element="all", path="backend/models/saves/" + self.__class__.__name__ + "-V" + str(self.auto_save_version) + ".auto-save.pth")

        return all_total_loss, all_contrastive_loss, all_classification_loss

    def _train_encoder(self, epochs: int, batch_size, games: list[Game], loader: Loader = None):
        """
        Train the model on the encoder head.
        """

        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.0005)

        num_samples = len(games or [])
        num_batches = (num_samples + batch_size - 1) // batch_size

        all_total_loss = []
        all_total_loss_one_hot = []
        all_total_loss_turn = []
        all_total_contrastive_loss = []

        for epoch in range(epochs):
            total_loss = 0
            total_loss_one_hot = 0
            total_loss_turn = 0
            total_contrastive_loss = 0

            if loader.need_update(epoch):
                games, _ = self._exctract_data(loader, epoch)
                num_samples = len(games)
                num_batches = (num_samples + batch_size - 1) // batch_size

            indices = torch.randperm(num_samples)

            for batch_idx in tqdm.tqdm(range(num_batches), ncols=TrainConfig.line_length+2, desc="| "):

                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_games = [games[i] for i in batch_indices]
                batch_games_t = [game.reverse() for game in batch_games]

                # predict the moves
                one_hot, turns, decoded_one_hot_logits, decoded_turns, latent = self.module(batch_games, head="encoder")
                _, _, _, _, latent_t = self.module(batch_games_t, head="encoder")

                # Compute loss
                decoded_one_hot_logits = decoded_one_hot_logits.view(-1, 12)
                one_hot = one_hot.view(-1, 12)  # Flatten one-hot target
                target_labels = one_hot.argmax(dim=-1)  # Shape: [batch_size]

                one_hot_target = torch.zeros_like(decoded_one_hot_logits)  # Create a tensor of zeros with the same shape as logits
                one_hot_target.scatter_(1, target_labels.view(-1, 1), 1)  # Fill the tensor with 1s at the target indices   
                
                loss_one_hot = F.binary_cross_entropy_with_logits(decoded_one_hot_logits, one_hot_target)
                loss_turn = F.binary_cross_entropy_with_logits(decoded_turns, turns)
                contrastive_loss = self._contrastive_loss(latent, latent_t) # fondamentally not totally correct cuz color inversion latent + color inversion = latent_t but we don't add the vector color inversion should be fixe

                loss = loss_one_hot + loss_turn + contrastive_loss * 0 # * 0.01 # increase when fixed

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_loss_one_hot += loss_one_hot.item()
                total_loss_turn += loss_turn.item()
                total_contrastive_loss += contrastive_loss.item()

            all_total_loss.append(total_loss / num_batches)
            all_total_loss_one_hot.append(total_loss_one_hot / num_batches)
            all_total_loss_turn.append(total_loss_turn / num_batches)
            all_total_contrastive_loss.append(total_contrastive_loss / num_batches)

            if self._train_config["UI"] == 'prints':
                l1 = f"  [{epoch + 1}/{epochs}] Loss: {total_loss / num_batches:.2f}, One-Hot Loss: {total_loss_one_hot / num_batches:.2f}, Turn Loss: {total_loss_turn / num_batches:.2f}"
                l2 = f"  Contrastive Loss: {total_contrastive_loss / num_batches:.2f}"
                l1 = Style("INFO", f"{l1: <{TrainConfig.line_length-2}}")
                l2 = Style("INFO", f"{l2: <{TrainConfig.line_length-2}}")
                print(f"| {l1} |")
                print(f"| {l2} |")
                print(f"| {'': <{TrainConfig.line_length-2}} |")

            if self._train_config["auto_save"]:
                if self.auto_save_version is None:
                    files = [f for f in os.listdir("backend/models/saves") if self.__class__.__name__ in f and "auto-save" in f]
                    self.auto_save_version = len(files)
                self.save(element="all", path="backend/models/saves/" + self.__class__.__name__ + "-V" + str(self.auto_save_version) + ".auto-save.pth")

        return all_total_loss
        
    def _train_on_generation(self, epochs: int, batch_size, games: list[Game], labels: Union[list[float], list[chess.Move]], loader: Loader = None):
        
        if loader is None:
            assert len(games) == len(labels) > 0, "You need at least one game to train, and one label per game."
            assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
            assert isinstance(labels[0], chess.Move), "'labels' should be a list of integers or instances of <chess.Move>."

        assert batch_size > 1, "Batch size must be greater than 1."

        if loader is not None:
            assert loader.window or 2 > 1, "Loader must have a window size greater than 1."

        # get moves dict (dict of all possible moves)
        moves_dict = None
        if os.path.exists("backend/data/moves_dict.pth"):
            moves_dict = torch.load("backend/data/moves_dict.pth", weights_only=False)
        elif os.path.exists(".data/moves_dict.pth"):
            moves_dict = torch.load(".data/moves_dict.pth", weights_only=False)
        else:
            moves_dict = {} # 64*64*5
            for i in range(64):
                for j in range(64):
                    # if i == j: continue: we keep total illegal move cuz we don't remove them in NN
                    for k in range(5):
                        # most of the time, promotion is not possible but we still considere the move, cuz easier
                        
                        moves_dict[(i, j, k)] = len(moves_dict)
            torch.save(moves_dict, "backend/data/moves_dict.pth")

        optimizer = torch.optim.Adam(self.module.parameters(), lr=0.0005)
        criterion = nn.CrossEntropyLoss()
        legal_criterion = nn.BCEWithLogitsLoss()

        num_samples = len(games or [])
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ensure full coverage

        all_total_loss = []
        all_legal_loss = []
        all_best_move_loss = []
        all_contrastive_loss = []

        for epoch in range(epochs):
            total_loss, total_best_move_loss, total_legal_loss, total_contrastive_loss = 0, 0, 0, 0

            if loader.need_update(epoch):
                games, best_moves = self._exctract_data(loader, epoch)

                num_samples = len(games)
                num_batches = (num_samples + batch_size - 1) // batch_size

            indices = torch.randperm(num_samples)

            for batch_idx in tqdm.tqdm(range(num_batches), ncols=TrainConfig.line_length+2, desc="| "):

                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_games = [games[i] for i in batch_indices]
                batch_best_moves = [best_moves[i] for i in batch_indices]
                batch_games_t = [game.reverse() for game in batch_games]
                batch_best_moves_t = [Game.reverse_move(move) for move in batch_best_moves]

                # predict the moves
                predictions, embeddings = self.module(batch_games, head="generation") # [batch_size, 64*64*5]
                predictions_t, embeddings_t = self.module(batch_games_t, head="generation")

                targets = [moves_dict[self.encode_move(move)] for move in batch_best_moves]
                targets = torch.tensor(targets, dtype=torch.long, device=self.module.device)

                targets_t = [moves_dict[self.encode_move(move)] for move in batch_best_moves_t]
                targets_t = torch.tensor(targets_t, dtype=torch.long, device=self.module.device)

                # Compute loss: 1 to increase the prob of the best move, 0 for the others
                best_move_loss = criterion(predictions, targets)
                best_move_t_loss = criterion(predictions_t, targets_t)
                best_move_loss = (best_move_loss + best_move_t_loss) / 2

                # Compute a loss about legals move (1 if legal 0 if not)
                all_legal_moves = [[moves_dict[(self.encode_move(move))] for move in game.board.legal_moves] for game in batch_games]
                legal_moves = [[int(i in legal_moves) for i in range(64*64*5)] for legal_moves in all_legal_moves]
                legal_moves = torch.tensor(legal_moves, dtype=torch.float, device=self.module.device)
                legal_loss = legal_criterion(predictions.float(), legal_moves.float())
                num_legal_moves = legal_moves.sum(dim=1, keepdim=True).clamp(min=1)
                legal_loss = (legal_loss / num_legal_moves).sum() * 0.01 # normalize by the number of legal moves

                # Compute contrastive loss
                contrastive_weight = 0.1 if epoch > 4 else 0 # warmup
                # contrastive loss have to be fixed: see train on encoder
                # contrastive_loss = self._contrastive_loss(embeddings, embeddings_t) * contrastive_weight

                # Backpropagation
                optimizer.zero_grad()
                loss = best_move_loss + legal_loss #+ contrastive_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_legal_loss += legal_loss.item()
                total_best_move_loss += best_move_loss.item()
                # total_contrastive_loss += contrastive_loss.item()

            all_total_loss.append(total_loss / num_batches)
            all_legal_loss.append(total_legal_loss / num_batches)
            all_best_move_loss.append(total_best_move_loss / num_batches)
            # all_contrastive_loss.append(total_contrastive_loss / num_batches)

            if self._train_config["UI"] == 'prints':
                l1 = f"  [{epoch + 1}/{epochs}] Loss: {total_loss / num_batches:.2f}, Best Move Loss: {total_best_move_loss / num_batches:.2f}, Contrastive loss {total_contrastive_loss / num_batches:.2f}" 
                l2 = f"  Legal Loss: {total_legal_loss / num_batches:.2f}"
                l1 = Style("INFO", f"{l1: <{TrainConfig.line_length-2}}")
                l2 = Style("INFO", f"{l2: <{TrainConfig.line_length-2}}")
                print(f"| {l1} |")
                print(f"| {l2} |")
                print(f"| {'': <{TrainConfig.line_length-2}} |")

            if self._train_config["auto_save"]:
                if self.auto_save_version is None:
                    files = [f for f in os.listdir("backend/models/saves") if self.__class__.__name__ in f and "auto-save" in f]
                    self.auto_save_version = len(files)
                self.save(element="all", path="backend/models/saves/" + self.__class__.__name__ + "-V" + str(self.auto_save_version) + ".auto-save.pth")

        return all_total_loss, all_legal_loss, all_best_move_loss

    def _test_generation(self, games: list[Game], best_moves: list[chess.Move], loader: Loader = None):

        if loader is not None:
            games, best_moves = self._exctract_data(loader, 0, _for="test")

        correct_predictions = 0
        total_games = len(games)

        moves_dict = None
        if os.path.exists("backend/data/moves_dict.pth"):
            moves_dict = torch.load("backend/data/moves_dict.pth", weights_only=False)
        elif os.path.exists(".data/moves_dict.pth"):
            moves_dict = torch.load(".data/moves_dict.pth", weights_only=False)
        else:
            # create moves dict
            moves_dict = {}
            for i in range(64):
                for j in range(64):
                    if i == j: continue
                    for k in range(5):
                        moves_dict[(i, j, k)] = len(moves_dict)
            torch.save(moves_dict, "backend/data/moves_dict.pth")

        self.module.eval()
        with torch.no_grad():
            predictions, _ = self.module(games, head="generation")
            predictions = torch.argmax(predictions, dim=1)

        for i, best_move in enumerate(best_moves):
            predicted_move = self.decode_move(predictions[i].item())
            correct_predictions += (predicted_move == best_move)

        if self._train_config["UI"] == 'prints':
            l = f"  Accuracy: {correct_predictions / total_games:.2f}"
            l = Style("SECONDARY_INFO", f"{l: <{TrainConfig.line_length-2}}")
            print(f"| {l} |")

        return correct_predictions / total_games

    def _test_encoder(self, games: list[Game], loader: Loader = None):
        
        if loader is not None:
            games, _ = self._exctract_data(loader, 0, _for="test")

        self.module.eval()
        with torch.no_grad():
            one_hot, turns, decoded_one_hot, decoded_turns, _ = self.module(games, head="encoder")

        # Compute accuracy
        correct_predictions = 0

        for i, game in enumerate(games):
            # 1. flatten the one_hot
            one_hot_i = one_hot[i].view(-1)
            decoded_one_hot_i_logits = decoded_one_hot[i].view(-1) # return logits

            # logits to one_hot
            max_indices = torch.argmax(decoded_one_hot_i_logits, dim=-1)  # Shape: (b, 8, 8, 13)
            decoded_one_hot_i = torch.zeros_like(decoded_one_hot_i_logits)
            decoded_one_hot_i.scatter_(-1, max_indices.unsqueeze(-1), 1)

            # 2. compare the one_hot and the decoded_one_hot
            correct_predictions += (one_hot_i == decoded_one_hot_i).sum().item()
            correct_predictions += (turns[i] == decoded_turns[i]).sum().item()

            # 3. normalize 8*8*12 + 1
            correct_predictions /= 8*8*12 + 1

        if self._train_config["UI"] == 'prints':
            l = f"  Accuracy: {correct_predictions / len(games):.2f}"
            l = Style("SECONDARY_INFO", f"{l: <{TrainConfig.line_length-2}}")
            print(f"| {l} |")

    def _test_board_evaluation(self, games: list[Game], win_probs: list[tuple[float, float]], loader: Loader = None):
        
        if loader is not None:
            games, win_probs = self._exctract_data(loader, 0, _for="test")

        self.module.eval()
        with torch.no_grad():
            probs, _ = self.module(games, head="board_evaluation")
            probs = torch.round(probs)

        # Compute accuracy
        correct_predictions = 0

        for i, game in enumerate(games):
            correct_predictions += (probs[i] == torch.tensor(win_probs[i], device=self.module.device)).all().item()

        if self._train_config["UI"] == 'prints':
            l = f"  Accuracy: {correct_predictions / len(games):.2f}"
            l = Style("SECONDARY_INFO", f"{l: <{TrainConfig.line_length-2}}")
            print(f"| {l} |")

    def _contrastive_loss(self, origin, rev):
        """
        Compute contrastive loss from two views (original and reversed games).
        origin: Tensor of shape [batch_size, embed_dim]
        rev: Tensor of shape [batch_size, embed_dim]
        """
        batch_size = origin.shape[0]
        device = origin.device

        # Normalize embeddings for cosine similarity
        origin = F.normalize(origin, dim=1)  # [batch_size, embed_dim]
        rev = F.normalize(rev, dim=1)  # [batch_size, embed_dim]

        # Compute cosine similarity between all (orig, rev) pairs
        similarity_matrix = torch.einsum("bd,cd->bc", origin, rev)  # [batch_size, batch_size]

        # Use a learnable temperature parameter (default: 0.07)
        if not hasattr(self, "temperature"):
            self.temperature = nn.Parameter(torch.tensor(0.07, device=device))
        
        logits = similarity_matrix / self.temperature  # Scale by temperature

        # Labels: Each sample i should match with sample i in rev (diagonal matches)
        labels = torch.arange(batch_size, device=device)

        # Compute contrastive loss (InfoNCE loss)
        loss = F.cross_entropy(logits, labels)

        return loss

    def predict(self, head="generation"):
        """
        Predict the best move for the current game.
        """
        self.module.eval()
        with torch.no_grad():
            scores, _ = self.module([self.game, self.game], head=head) # 2 views of the same game for batchnorm
        if head == "generation":
            return scores.tolist()[0]
        else:
            return scores.tolist()[0]

# Instances des configurations (corrige le problème de coloration syntaxique)
all_heads = AllHeads()
"""[configuration]: Train the model on all heads."""

generative_head = GenerativeHead()
"""[configuration]: Train the model on the generative head."""

board_evaluation_head = BoardEvaluationHead()
"""[configuration]: Train the model on the board evaluation head."""

encoder_head = EncoderHead()
"""[configuration]: Train the model on the encoder head."""

with_prints = WithPrints()
"""[configuration]: Print the training steps."""

auto_save = AutoSave()
"""[configuration]: Automatically save the model."""

class DeepEngineModule(nn.Module):

    def __init__(self):
        super(DeepEngineModule, self).__init__()

        self.classifier = DefaultClassifier()
        """torch.nn.Module: The classifier head of the model. Should return [batch_size, 2]. -> [black_win, white_win]"""

        self.generative_head = DefaultGenerativeHead()
        """torch.nn.Module: The generative head of the model. Should return [batch_size, 64*64*5]."""

        self.embedding = Embedding()
        """
            torch.nn.Module: The embedding head of the model. 
            Should return [batch_size, latent_dim].
            Take in input [batch_size, 8, 8, 14] (one-hot encoding of the game + turn projection)
        """

        self.decoder = DefaultDecoder()
        """torch.nn.Module: The decoder head of the model. Should return [batch_size, 8, 8, 12]."""

        self.turn_projection = nn.Linear(1, 8*8)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.set_device()

    def set_device(self):
        self.to(self.device)
        self.turn_projection.to(self.device)
        self.classifier.to(self.device)
        self.embedding.to(self.device)

    def encode_games(self, one_hots, turns) -> torch.Tensor:
        """
        Encode the one-hot representation of the games.
        
        :param one_hots: one-hot representation of the games
        :type one_hots: torch.Tensor
        :param turns: turn of the games
        :type turns: torch.Tensor

        :return: encoded games: [batch_size, 8, 8, 13+1]
        :rtype: torch.Tensor
        """

        projected_turns: torch.Tensor = self.turn_projection(turns)
        projected_turns = projected_turns.reshape(turns.shape[0], 8, 8, 1)

        # shape are [batch_size, 8, 8, 1] and [batch_size, 8, 8, 13]
        # we want to concatenate them along the last dimension: [batch_size, 8, 8, 14]
        encoding = torch.cat([one_hots, projected_turns], dim=-1)
        encoding = encoding.to(dtype=torch.float, device=self.device)
        return encoding


    def forward(self, games: list[Game], head="board_evaluation") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the score of a game for the given color.

        :param games: list of games
        :type games: list[Game]

        :param head: head to use for the inference
        :type head: str

        :return: score of the game
        :rtype: torch.Tensor of shape [batch_size, 2]
        :return: embedding of the game
        :rtype: torch.Tensor of shape [batch_size, latent_dim]
        """

        one_hots = torch.stack([torch.tensor(game.one_hot(), dtype=torch.float, device=self.device) for game in games])
        turns = torch.tensor([[int(game.board.turn)] for game in games], dtype=torch.float, device=self.device)
        encoding = self.encode_games(one_hots, turns) # [batch_size, 8, 8, 14]

        embedding = self.embedding(encoding) # [batch_size, latent_dim]

        if head == "board_evaluation":
            scores = self.classifier(embedding)
        elif head == "generation":
            scores = self.generative_head(embedding)
        elif head == "encoder":
            decoded_one_hots, decoded_turns = self.decoder(embedding)
            if isinstance(embedding, tuple):
                return one_hots, turns, decoded_one_hots, decoded_turns, embedding[0]
            return one_hots, turns, decoded_one_hots, decoded_turns, embedding
        else:
            raise ValueError(f"Invalid head: {head}")
        
        if isinstance(embedding, tuple): # allow to work with rich embeddings
            return scores, embedding[0]

        return scores, embedding  
        
