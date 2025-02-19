from src.chess.game import Game
from models.engine import Engine

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEngine(Engine):
    """
    Implementation of a simple AI that plays random moves.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "Simple AI that plays random moves."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_function = ScoreModel()

    def play(self, **kwargs):
        """
        Play a random move.
        """

        legal_moves = list(self.game.board.legal_moves)
        if not legal_moves:
            return None
        
        scores = {}
        for move in legal_moves:
            self.game.board.push(move)
            with torch.no_grad():
                scores[move] = self.score_function(self.game, self.color)
            self.game.board.pop()

        best_move = max(scores, key=scores.get)
        return best_move

    def train(self, games: list[Game], best_moves: list[chess.Move], epochs: int):
        """
        Use contrastive learning:
            - x: game
            - x_t: mirrored game
            - y: best move: 1 move per game
        """

        assert len(games) == len(best_moves) > 0, "You need at least one game to train, and you need one move per game"
        assert type(games[0]) is Game, "'games' should be a list of instance of <Game>"
        assert type(best_moves[0]) is chess.Move, "'moves' should be a list of instance of <chess.Move>"

        classifier_optimizer = torch.optim.Adam(self.score_function.classifier.parameters(), lr=0.001)
        embedding_optimizer = torch.optim.Adam(self.score_function.embedding.parameters(), lr=0.0005)  # Lower LR

        criterion = nn.CrossEntropyLoss()

        for _ in range(epochs):
            total_loss = 0
            contrastive_loss_total = 0
            embedding_loss_total = 0

            for idx, game in enumerate(games):
                legal_moves = list(game.board.legal_moves)
                if not legal_moves: continue

                best_move = best_moves[idx]
                game_t = game.reverse()

                scores, scores_t = {}, {}
                embeddings, embeddings_t = {}, {}

                for move in legal_moves:
                    game.board.push(move)
                    scores[move], embeddings[move] = self.score_function([game], self.color)
                    game.board.pop()

                    game_t.board.push(move)
                    scores_t[move], embeddings_t[move] = self.score_function([game_t], not self.color)
                    game_t.board.pop()

                # Convert to tensors
                scores_tensor = torch.tensor([scores[m] for m in legal_moves], dtype=torch.float).unsqueeze(0)
                scores_t_tensor = torch.tensor([scores_t[m] for m in legal_moves], dtype=torch.float).unsqueeze(0)

                embeddings_tensor = torch.stack([embeddings[m] for m in legal_moves])  # Shape: (num_moves, embedding_dim)
                embeddings_t_tensor = torch.stack([embeddings_t[m] for m in legal_moves])  # Mirrored embeddings

                y = torch.tensor([int(move == best_move) for move in legal_moves], dtype=torch.float).unsqueeze(0)

                # 1. Classification loss (choosing the best move)
                classification_loss = criterion(scores_tensor, y)

                # 2. Contrastive loss (score consistency for mirrored positions)
                contrastive_loss = F.mse_loss(scores_tensor, scores_t_tensor)

                # 3. Embedding similarity loss (L2 distance between embeddings and their mirrored counterparts)
                embedding_loss = F.mse_loss(embeddings_tensor, embeddings_t_tensor)

                total_loss += classification_loss.item()
                contrastive_loss_total += contrastive_loss.item() * 0.5
                embedding_loss_total += embedding_loss.item() * 0.5

                # Backpropagation
                classifier_optimizer.zero_grad()
                classification_loss.backward()
                classifier_optimizer.step()

                embedding_optimizer.zero_grad()
                (contrastive_loss + embedding_loss).backward()
                embedding_optimizer.step()

            print(f"[{_+1}/{epochs}] Loss: {total_loss / len(games)}, Contrastive Loss: {contrastive_loss_total / len(games)}, Embedding Loss: {embedding_loss_total / len(games)}")

    def evaluate(self, games, best_moves):

        assert len(games) == len(best_moves) > 0, "You need at least one game to train, and you need one move per game"
        assert type(games[0]) is Game, "'games' should be a list of instance of <Game>"
        assert type(best_moves[0]) is chess.Move, "'moves' should be a list of instance of <chess.Move>"

        total_acc = 0

        for idx, game in enumerate(games):
            legal_moves = list(game.board.legal_moves)
            if not legal_moves: continue

            best_move = best_moves[idx]

            scores = {}
            embeddings = {}

            for move in legal_moves:
                game.board.push(move)
                with torch.no_grad(): scores[move], embeddings[move] = self.score_function([game], self.color)
                game.board.pop()

            predicted_move = max(scores, key=scores.get)
            total_acc += predicted_move == best_move

        return total_acc / len(games)


class ScoreModel(nn.Module):

    def __init__(self):
        super(ScoreModel, self).__init__()

        self.classifier = TestClassifier()
        """torch.nn.Module: The classifier head of the model. Should return [batch_size, 2]. -> [black_win, white_win]"""

        self.embedding = TestEmbedding()
        """
            torch.nn.Module: The embedding head of the model. 
            Should return [batch_size, latent_dim].
            Take in 
        """

        self.turn_projection = nn.Linear(1, 8*8)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

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

        projected_turns = self.turn_projection(turns)
        projected_turns = projected_turns.reshape(1, 8, 8).unsqueeze(0)
        projected_turns.expand(one_hots.shape[0], 1, 8, 8, dim=3)
        print(projected_turns.shape)
        encoding = torch.concat([one_hots, projected_turns])
        print(encoding)
        return torch.stack(encoding)


    def forward(self, games: list[Game], color) -> float:
        """
        Compute the score of a game for the given color.
        """

        one_hots = torch.stack([torch.tensor(game.one_hot(), dtype=torch.float) for game in games])
        turns = torch.tensor([int(game.board.turn) for game in games], dtype=torch.float)
       
        encoding = self.encode_games(one_hots, turns) # [batch_size, 8, 8, 14]
        encoding = encoding.to(self.device)

        embedding = self.embedding(encoding) # [batch_size, latent_dim]
        scores = self.classifier(embedding) # [batch_size, 2]

        return scores, embedding        
        
class TestEmbedding(nn.Module):

    def __init__(self):
        super(TestEmbedding, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=14, out_channels=64, kernel_size=8) # [b, 8, 8 14] -> [b, 1, 1, 64]

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)
        return x
    
class TestClassifier(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(64, 2)
    
    def forward(self, x):
        return self.fc1(x)
