from functools import wraps
from typing import Union
from src.chess.game import Game
from models.engine import Engine
from models.train_config import *
from models.template import DefaultClassifier, Embedding

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_function = ScoreModel()
        """torch.nn.Module: The model used to score the games."""

        self.is_loaded = False
        self._train_config = {"mode": None, "head": None, "UI": None, "auto_save": False}


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

        legal_moves = list(self.game.board.legal_moves)
        if not legal_moves:
            return None
        
        scores = {}
        for move in legal_moves:
            self.game.board.push(move)
            with torch.no_grad():
                scores[move], _ = self.score_function([self.game])
            self.game.board.pop()

            x = scores[move].tolist()[0]
            scores[move] = x[int(self.color)]
            print(f"Move: {move} - Score: {scores[move]} || {x[int(not self.color)]}")

        best_move = max(scores, key=scores.get)
        # for m, s in scores.items():
        #     print(f"{m}: {s}", "  <---" if m == best_move else "")
        return best_move
    
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

    def info_nce_loss(self, features, temperature=0.1):
        """
        Compute the InfoNCE loss for contrastive learning.
        
        :param features: Tensor of shape (batch_size, embedding_dim)
        :param temperature: Temperature parameter for scaling
        :return: loss value
        """

        batch_size = features.shape[0]
        device = features.device

        # Normalize embeddings
        features = F.normalize(features, dim=1)  

        # Compute similarity matrix (cosine similarity)
        similarity_matrix = torch.matmul(features, features.T)

        # Create labels (identity matrix for positive pairs)
        labels = torch.eye(batch_size, dtype=torch.float, device=device)

        # Mask out self-comparisons
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        similarity_matrix = similarity_matrix[~mask].view(batch_size, -1)
        labels = labels[~mask].view(batch_size, -1)

        # Separate positives and negatives
        positives = similarity_matrix[labels.bool()].view(batch_size, -1)
        negatives = similarity_matrix[~labels.bool()].view(batch_size, -1)

        # Concatenate positives and negatives (positives first)
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=device)

        # Apply temperature scaling
        # logits = logits / temperature

        # Compute loss
        return F.cross_entropy(logits, labels)

    # train mode:
    # 1. from puzzle (cleanest way: cuz puzzle is perfect move)
    # 2. from game (more data, but less perfect, however can understand more commun situation and better understand of loosing)
    # 3. Auto encoder (board to board, improve the embedding, but not able to predict the best move)

    # other idears:
    # 1. getting the score for white and black (understand the position, basic value of pieces, but not meaningfull)
    # 2. getting the list of legal moves (understand the position / rules, but not linked to the best move, only game rules / representation)
    # 3. generating the best move (over hard: generation have to respect rules etc, less safe than just giving scores) ??
    # 4. get nb attacks per square (understand the position / turn notion, weak learning)

    def train2(self, games: list[Game], best_moves: list[chess.Move], epochs: int, batch_size: int = 32):
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

        for epoch in range(epochs):
            total_loss = 0
            contrastive_loss_total = 0
            embedding_loss_total = 0
            classification_loss_total = 0

            indices = torch.randperm(num_samples)  # Shuffle indices
            for batch_idx in tqdm.tqdm(
                range(num_batches), 
                desc=f"{num_batches} batches of {batch_size} games of x moves"
            ):
                
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

            print(f"|___ [{epoch + 1}/{epochs}] Loss: {total_loss / num_batches:.2f}, Contrastive Loss: {contrastive_loss_total / num_batches:.2f}, Embedding Loss: {embedding_loss_total / num_batches:.2f} Classification Loss: {classification_loss_total / num_batches:.2f}\n")

        return all_total_loss, all_contrastive_loss, all_embedding_loss, all_classification_loss
    
    def embedding_contrastive_loss(self, emb_orig, emb_rev):
        """
        Compute InfoNCE-style contrastive loss for embeddings from two views.
        emb_orig, emb_rev: Tensors of shape [n, d] for n candidate moves.
        For each candidate in emb_orig, its positive is the corresponding candidate in emb_rev,
        and vice-versa.
        """
        n = emb_orig.shape[0]
        device = emb_orig.device
        # Concatenate to get a [2*n, d] tensor.
        features = torch.cat([emb_orig, emb_rev], dim=0)
        features = F.normalize(features, dim=1)
        
        # Compute cosine similarity matrix and scale by temperature.
        temperature = self.temperature if hasattr(self, "temperature") else 0.1
        logits = torch.matmul(features, features.T) / temperature  # Shape: [2*n, 2*n]
        
        # Mask out self-similarity by setting diagonal to a very low value.
        mask = torch.eye(2 * n, dtype=torch.bool, device=device)
        logits = logits.masked_fill(mask, -1e9)
        
        # For each sample i in 0..2*n-1, its positive pair is at index:
        # i + n if i < n, else i - n.
        labels = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)], dim=0).to(device)
        
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
        for game, best_move in tqdm.tqdm(zip(games, best_moves), desc=f"Testing {total_games} games", total=total_games):
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

        return correct_predictions / total_games

    def __or__(self, other: TrainConfigBase) -> TrainConfig:
        """Permet d'enchaîner les configurations avec `|` et retourne le bon type."""
        return other.apply(self)

    def train(self, **data):
        """Exécute l'entraînement."""
        mode = self._train_config["mode"]
        head = self._train_config["head"]
        print(f"🚀 Training in mode '{mode}' with head '{head}'...")
        print("✨ Magic happens here ✨")

    def test(self, _plot=False):
        """Évaluation du modèle."""
        print(f"📊 Evaluating... (plot={_plot})")
        print("📈 Results generated.")


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


    def forward(self, games: list[Game]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the score of a game for the given color.

        :param games: list of games
        :type games: list[Game]

        :return: score of the game
        :rtype: torch.Tensor of shape [batch_size, 2]
        :return: embedding of the game
        :rtype: torch.Tensor of shape [batch_size, latent_dim]
        """

        one_hots = torch.stack([torch.tensor(game.one_hot(), dtype=torch.float, device=self.device) for game in games])
        turns = torch.tensor([[int(game.board.turn)] for game in games], dtype=torch.float, device=self.device)
        encoding = self.encode_games(one_hots, turns) # [batch_size, 8, 8, 14]

        embedding = self.embedding(encoding) # [batch_size, latent_dim]
        scores = self.classifier(embedding) # [batch_size, 2]

        return scores, embedding  
        
