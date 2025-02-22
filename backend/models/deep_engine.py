from functools import wraps
from typing import Union
from src.utils.console import Style
from src.chess.game import Game
from models.engine import Engine
from models.train_config import *
from models.template import DefaultClassifier, Embedding, DefaultGenerativeHead

import chess
import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DeepEngine", "all_heads", "on_puzzles", "on_games", "generative_head", "board_evaluation_head", "encoder_head", "with_prints", "auto_save"]

class DeepEngine(Engine):
    """
    Implementation of a simple AI that plays random moves.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "Simple AI that plays random moves."

    PROMOTION_TABLE = {
        None: 0,
        chess.QUEEN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_function = ScoreModel()
        """torch.nn.Module: The model used to score the games."""

        self.is_loaded = False
        self.auto_save_version = None
        self._train_config = {"mode": None, "head": None, "UI": None, "auto_save": False}

    def __or__(self, other: TrainConfigBase) -> TrainConfig:
        """Permet d'enchaîner les configurations avec `|` et retourne le bon type."""
        return other.apply(self)

    def play(self, **kwargs):
        """
        Play a random move.
        """

        if not self.is_loaded:
            path = "./models/saves/" + self.__class__.__name__ + ".active.pth"
            if os.path.exists(path):
                self.load(path)
                self.is_loaded = True
            else:
                raise FileNotFoundError(f"Model not found at {path}")

        scores = self.predict()
        legal_moves = list(self.game.board.legal_moves)
        scores = [scores[self.encode_move(move, as_int=True)] for move in legal_moves]
        return legal_moves[scores.index(max(scores))]

    
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
            self.score_function.load_state_dict(torch.load(path, weights_only=True))
        elif element == "embedding":
            self.score_function.embedding.load_state_dict(torch.load(path, weights_only=True))
        elif element == "classifier":
            self.score_function.classifier.load_state_dict(torch.load(path, weights_only=True))

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
            torch.save(self.score_function.state_dict(), path)
        elif element == "embedding":
            torch.save(self.score_function.embedding.state_dict(), path)
        elif element == "classifier":
            torch.save(self.score_function.classifier.state_dict(), path)

    def _train_on_puzzles(self, epochs: int, batch_size, **data):
        """
        Train the model on puzzles.
        """
        games = data.get("games", None)
        best_moves = data.get("moves", None) or data.get("best_moves", None)

        assert games is not None, "You need to provide games to train on."
        assert best_moves is not None, "You need to provide best moves to train on."

        if self._train_config["head"] == "all":
            ...
        elif self._train_config["head"] == "generative":
            ...
        elif self._train_config["head"] == "board_evaluation":
            return self._train_board_evaluation(epochs, batch_size, games, best_moves, format="move")
        elif self._train_config["head"] == "encoder":
            ...
        else:
            raise ValueError(f"Invalid head: {self._train_config['head']}")
        
    def _train_board_evaluation(self, epochs: int, batch_size, games: list[Game], labels: Union[list[int], list[chess.Move]], format):
        """
        Train the model on the board evaluation head.

        :param epochs: number of epochs
        :type epochs: int

        :param batch_size: size of the batch
        :type batch_size: int

        :param games: list of games
        :type games: list[Game]

        :param labels: list of labels -> the list of int (0 black win, 1 white win) OR the list of best moves (1 per game)
        :type labels: Union[list[int], list[chess.Move]]

        :param format: format of the labels (either "int" or "move")
        :type format: str
        """

        assert len(games) == len(labels) > 0, "You need at least one game to train, and one label per game."
        assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
        assert isinstance(labels[0], (int, chess.Move)), "'labels' should be a list of integers or instances of <chess.Move>."
        assert format in ["int", "move"], "Invalid format for labels, should be either 'int' or 'move'."

        if format == "move":
            self._train_board_evaluation_puzzles(games, labels, epochs, batch_size)
        else:
            raise NotImplementedError("Training on board evaluation with integer labels is not implemented yet.")

    def _train_board_evaluation_puzzles(self, games: list[Game], best_moves: list[chess.Move], epochs: int, batch_size: int = 32):
        """
        Train the model using batches instead of processing one element at a time.

        :param games: list of games
        :type games: list[Game]

        :param best_moves: list of best moves
        :type best_moves: list[chess.Move]

        :param epochs: number of epochs
        :type epochs: int

        :param batch_size: size of the batch
        :type batch_size: int

        :return: total_loss, contrastive_loss, embedding_loss, classification_loss
        :rtype: tuple[list[float], list[float], list[float], list[float]]
        """

        assert len(games) == len(best_moves) > 0, "You need at least one game to train, and one move per game."
        assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
        assert isinstance(best_moves[0], chess.Move), "'moves' should be a list of instances of <chess.Move>."

        # classifier_optimizer = torch.optim.Adam(self.score_function.classifier.parameters(), lr=0.001)
        # embedding_optimizer = torch.optim.Adam(self.score_function.embedding.parameters(), lr=0.0005)  # Lower LR
        optimizer = torch.optim.Adam(self.score_function.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        num_samples = len(games)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ensure full coverage

        all_total_loss = []
        all_contrastive_loss = []
        all_embedding_loss = []
        all_classification_loss = []

        # if self._train_config["UI"] == 'prints':
        #     print(f"\t {num_batches} batches of {batch_size} games of x moves each.")

        for epoch in range(epochs):
            total_loss = 0
            contrastive_loss_total = 0
            embedding_loss_total = 0
            classification_loss_total = 0

            indices = torch.randperm(num_samples)  # Shuffle indices
            for batch_idx in tqdm.tqdm(range(num_batches), ncols=TrainConfig.line_length+2, desc="| "):
                
                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_games = [games[i] for i in batch_indices]
                batch_best_moves = [best_moves[i] for i in batch_indices]

                # Collect all positions and targets for the batch
                for game, best_move in zip(batch_games, batch_best_moves):
                    legal_moves = list(game.board.legal_moves)
                    if not legal_moves:
                        continue

                    game_t = game.reverse()
                    game_positions = []
                    game_positions_t = []

                    targets = []
                    turns = []
                    turns_t = []

                    for move in legal_moves:
                        game.board.push(move)
                        game_positions.append(game.copy())
                        turns.append(game.board.turn)
                        game.board.pop()

                        targets.append(move == best_move)

                    for move in game_t.board.legal_moves:
                        game_t.board.push(move)
                        game_positions_t.append(game_t.copy())
                        turns_t.append(game_t.board.turn)
                        game_t.board.pop()

                    if not game_positions: continue

                    # model inference (per game: slow but allow crossentropy loss)
                    self.score_function.train()
                    scores, embeddings = self.score_function(game_positions)
                    scores_t, embeddings_t = self.score_function(game_positions_t)

                    # Extract scores dynamically based on the tracked player turns
                    turns_tensor = torch.tensor(turns, dtype=torch.long, device=self.score_function.device)
                    turns_t_tensor = torch.tensor(turns_t, dtype=torch.long, device=self.score_function.device)
                    scores = scores.gather(1, turns_tensor.view(-1, 1)).squeeze(1)  # Extract correct player's score
                    scores_t = scores_t.gather(1, turns_t_tensor.view(-1, 1)).squeeze(1)

                    # Convert to tensors
                    targets_tensor = torch.tensor(targets, dtype=torch.float, device=self.score_function.device)

                    # **Loss Computation**
                    classification_loss = criterion(scores, targets_tensor)
                    contrastive_loss = F.mse_loss(scores, scores_t)
                    embedding_loss = self.embedding_contrastive_loss(embeddings, embeddings_t)

                    # **Backpropagation**
                    optimizer.zero_grad()
                    loss = contrastive_loss + embedding_loss + classification_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    contrastive_loss_total += contrastive_loss.item()
                    embedding_loss_total += embedding_loss.item()
                    classification_loss_total += classification_loss.item()

            all_total_loss.append(total_loss / num_batches)
            all_contrastive_loss.append(contrastive_loss_total / num_batches)
            all_embedding_loss.append(embedding_loss_total / num_batches)
            all_classification_loss.append(classification_loss_total / num_batches)

            if self._train_config["UI"] == 'prints':
                l1 = f"  [{epoch + 1}/{epochs}] Loss: {total_loss / num_batches:.2f}, Contrastive Loss: {contrastive_loss_total / num_batches:.2f}, " 
                l2 = f"  Embedding Loss: {embedding_loss_total / num_batches:.2f} Classification Loss: {classification_loss_total / num_batches:.2f}"
                l1 = Style("INFO", f"{l1: <{TrainConfig.line_length-2}}")
                l2 = Style("INFO", f"{l2: <{TrainConfig.line_length-2}}")
                print(f"| {l1} |")
                print(f"| {l2} |")
                print(f"| {'': <{TrainConfig.line_length-2}} |")

        return all_total_loss, all_contrastive_loss, all_embedding_loss, all_classification_loss

    def _train_on_games(self, epochs: int, batch_size, **data):
        """
        Train the model on games.
        """

        games = data.get("games", None)
        labels = data.get("moves", None) or data.get("best_moves", None) or data.get("win_probs", None)

        assert games is not None, "You need to provide games to train on."
        assert labels is not None, "You need to provide labels (best moves or win probabilities) to train on."
        assert len(games) == len(labels) > 0, "You need at least one game to train, and one label per game."

        if self._train_config["head"] == "all":
            ...
        elif self._train_config["head"] == "generative":
            return self._train_on_generation(epochs, batch_size, games, labels, format="move")
        elif self._train_config["head"] == "board_evaluation":
            return self._train_board_evaluation(epochs, batch_size, games, labels, format="move")
        elif self._train_config["head"] == "encoder":
            ...
        else:
            raise ValueError(f"Invalid head: {self._train_config['head']}")
        
    def _train_on_generation(self, epochs: int, batch_size, games: list[Game], labels: Union[list[float], list[chess.Move]], format):

        assert len(games) == len(labels) > 0, "You need at least one game to train, and one label per game."
        assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
        assert isinstance(labels[0], (int, chess.Move)), "'labels' should be a list of integers or instances of <chess.Move>."
        assert format in ["float", "move"], "Invalid format for labels, should be either 'float' or 'move'."

        if format == "move":
            self._train_generation_moves(games, labels, epochs, batch_size)
        else:
            raise NotImplementedError("Training on generation with integer labels is not implemented yet.")

    def _train_generation_moves(self, games: list[Game], best_moves: list[chess.Move], epochs: int, batch_size: int = 32):
        """
        """

        assert batch_size > 1, "Batch size must be greater than 1."

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

        optimizer = torch.optim.Adam(self.score_function.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        legal_criterion = nn.BCEWithLogitsLoss()

        num_samples = len(games)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ensure full coverage

        all_total_loss = []
        all_legal_loss = []
        all_best_move_loss = []
        all_contrastive_loss = []

        for epoch in range(epochs):
            total_loss, total_best_move_loss, total_legal_loss, total_contrastive_loss = 0, 0, 0, 0
            indices = torch.randperm(num_samples)

            for batch_idx in tqdm.tqdm(range(num_batches), ncols=TrainConfig.line_length+2, desc="| "):

                batch_indices = indices[batch_idx * batch_size : (batch_idx + 1) * batch_size]
                batch_games = [games[i] for i in batch_indices]
                batch_best_moves = [best_moves[i] for i in batch_indices]
                batch_games_t = [game.reverse() for game in batch_games]
                batch_best_moves_t = [Game.reverse_move(move) for move in batch_best_moves]

                # predict the moves
                predictions, embeddings = self.score_function(batch_games, head="generation") # [batch_size, 64*64*5]
                predictions_t, embeddings_t = self.score_function(batch_games_t, head="generation")

                targets = [moves_dict[self.encode_move(move)] for move in batch_best_moves]
                targets = torch.tensor(targets, dtype=torch.long, device=self.score_function.device)

                # Compute loss: 1 to increase the prob of the best move, 0 for the others
                best_move_loss = criterion(predictions, targets)

                # Compute a loss about legals move (1 if legal 0 if not)
                legal_moves = [[moves_dict[(self.encode_move(move))] for move in game.board.legal_moves] for game in batch_games]
                legal_moves = [[1 if i in t else 0 for i in range(len(moves_dict))] for t in legal_moves]
                legal_moves = torch.tensor(legal_moves, dtype=torch.float, device=self.score_function.device)
                legal_loss = legal_criterion(predictions.float(), legal_moves.float())
                num_legal_moves = legal_moves.sum(dim=1, keepdim=True).clamp(min=1)
                legal_loss = (legal_loss / num_legal_moves).sum() * 0.01 # normalize by the number of legal moves

                # Compute contrastive loss
                contrastive_weight = 0.1 if epoch > 4 else 0 # warmup
                contrastive_loss = self.embedding_contrastive_loss(embeddings, embeddings_t) * contrastive_weight

                # Backpropagation
                optimizer.zero_grad()
                loss = best_move_loss + legal_loss + contrastive_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_legal_loss += legal_loss.item()
                total_best_move_loss += best_move_loss.item()
                total_contrastive_loss += contrastive_loss.item()

            all_total_loss.append(total_loss / num_batches)
            all_legal_loss.append(total_legal_loss / num_batches)
            all_best_move_loss.append(total_best_move_loss / num_batches)
            all_contrastive_loss.append(total_contrastive_loss / num_batches)

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

    def _test_on_games(self, _plot=False, **data):
        
        games = data.get("games", None)
        best_moves = data.get("best_moves", None) or data.get("moves", None)

        assert games is not None, "You need to provide games to test on."
        assert best_moves is not None, "You need to provide best moves to test on."

        return self._test_generation(games, best_moves)

    def _test_generation(self, games: list[Game], best_moves: list[chess.Move]):

        assert len(games) == len(best_moves) > 0, "You need at least one game and one move per game."
        assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
        assert isinstance(best_moves[0], chess.Move), "'moves' should be a list of instances of <chess.Move>."

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

        self.score_function.eval()
        with torch.no_grad():
            predictions, _ = self.score_function(games, head="generation")
            predictions = torch.argmax(predictions, dim=1)

        for i, best_move in enumerate(best_moves):
            predicted_move = self.decode_move(predictions[i].item())
            correct_predictions += (predicted_move == best_move)

        if self._train_config["UI"] == 'prints':
            l = f"  Accuracy: {correct_predictions / total_games:.2f}"
            l = Style("SECONDARY_INFO", f"{l: <{TrainConfig.line_length-2}}")
            print(f"| {l} |")

        return correct_predictions / total_games

    def embedding_contrastive_loss(self, emb_orig, emb_rev):
        """
        Compute contrastive loss for embeddings from two views (original and reversed games).
        emb_orig: Tensor of shape [batch_size, embed_dim]
        emb_rev: Tensor of shape [batch_size, embed_dim]
        """
        batch_size = emb_orig.shape[0]
        device = emb_orig.device

        # Normalize embeddings for cosine similarity
        emb_orig = F.normalize(emb_orig, dim=1)  # [batch_size, embed_dim]
        emb_rev = F.normalize(emb_rev, dim=1)  # [batch_size, embed_dim]

        # Compute cosine similarity between all (orig, rev) pairs
        similarity_matrix = torch.einsum("bd,cd->bc", emb_orig, emb_rev)  # [batch_size, batch_size]

        # Use a learnable temperature parameter (default: 0.07)
        if not hasattr(self, "temperature"):
            self.temperature = nn.Parameter(torch.tensor(0.07, device=device))
        
        logits = similarity_matrix / self.temperature  # Scale by temperature

        # Labels: Each sample i should match with sample i in rev (diagonal matches)
        labels = torch.arange(batch_size, device=device)

        # Compute contrastive loss (InfoNCE loss)
        loss = F.cross_entropy(logits, labels)

        return loss

    def test2(self, games: list[Game], best_moves: list[chess.Move]):
        """
        Evaluate the model's accuracy in predicting the best move.
        """
        
        assert len(games) == len(best_moves) > 0, "You need at least one game and one move per game."
        assert isinstance(games[0], Game), "'games' should be a list of instances of <Game>."
        assert isinstance(best_moves[0], chess.Move), "'moves' should be a list of instances of <chess.Move>."

        correct_predictions = 0
        total_games = len(games)

        all_moves = []
        all_game_positions = []
        all_game_positions_t = []
        best_moves_dict = {}
        turns = []
        turns_t = []

        # Collect all positions and targets for batch inference
        for game, best_move in tqdm.tqdm(zip(games, best_moves), desc=f"| ", total=total_games, ncols=TrainConfig.line_length+2):
            legal_moves = list(game.board.legal_moves)
            if not legal_moves:
                continue
            
            best_moves_dict[game] = best_move
            game_t = game.reverse()

            for move in legal_moves:
                game.board.push(move)
                all_game_positions.append(game.copy())
                turns.append(game.board.turn)
                game.board.pop()

                all_moves.append((game, move))  # Keep track of which move belongs to which game

            for move in game_t.board.legal_moves:
                game_t.board.push(move)
                all_game_positions_t.append(game_t.copy())
                turns_t.append(game_t.board.turn)
                game_t.board.pop()

        if not all_game_positions: return -1
        
        # **Batch model inference**
        self.score_function.eval()
        with torch.no_grad():
            scores, _ = self.score_function(all_game_positions)
            scores_t, _ = self.score_function(all_game_positions_t)

        # Extract scores dynamically based on tracked player turns
        turns_tensor = torch.tensor(turns, dtype=torch.long, device=self.score_function.device)
        turns_t_tensor = torch.tensor(turns_t, dtype=torch.long, device=self.score_function.device)
        scores = scores.gather(1, turns_tensor.view(-1, 1)).squeeze(1)  
        scores_t = scores_t.gather(1, turns_t_tensor.view(-1, 1)).squeeze(1)

        # Match scores back to games
        move_scores = {}
        for i, (game, move) in enumerate(all_moves):
            if game not in move_scores:
                move_scores[game] = {}
            move_scores[game][move] = scores[i].item()  # Extract scalar
        
        # Compute accuracy
        for game in games:
            if game in move_scores:
                best_move = best_moves_dict[game]
                predicted_move = max(move_scores[game], key=move_scores[game].get)
                correct_predictions += (predicted_move == best_move)

        if self._train_config["UI"] == 'prints':
            l = f"  Accuracy: {correct_predictions / total_games:.2f}"
            l = Style("SECONDARY_INFO", f"{l: <{TrainConfig.line_length-2}}")
            print(f"| {l} |")

        return correct_predictions / total_games

    def _test_on_puzzles(self, _plot=False, **data):
        """Évaluation du modèle."""
       
        games = data.get("games", None)
        best_moves = data.get("best_moves", None) or data.get("moves", None)

        assert games is not None, "You need to provide games to test on."
        assert best_moves is not None, "You need to provide best moves to test on."

        return self.test2(games, best_moves)

    def predict(self, head="generation"):
        """
        Predict the best move for the current game.
        """

        scores, _ = self.score_function([self.game], head=head)
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

on_puzzles = OnPuzzles()
"""[configuration]: Train the model on puzzles."""

on_games = OnGames()
"""[configuration]: Train the model on games."""

with_prints = WithPrints()
"""[configuration]: Print the training steps."""

auto_save = AutoSave()
"""[configuration]: Automatically save the model."""

class ScoreModel(nn.Module):

    def __init__(self):
        super(ScoreModel, self).__init__()

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


    def forward(self, games: list[Game], head="win_probs") -> tuple[torch.Tensor, torch.Tensor]:
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

        if head == "win_probs":
            scores = self.classifier(embedding)
        elif head == "generation":
            scores = self.generative_head(embedding)
        else:
            raise ValueError(f"Invalid head: {head}")

        return scores, embedding  
        
