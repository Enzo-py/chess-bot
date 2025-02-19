import numpy as np
from models.engine import Engine
from models.cnn.cnn_toolbox import ResBlock, PositionalEncoding, CBAMChannelAttention

from src.chess.game import Game

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNScore(Engine, nn.Module):
    """
    CNN-based AI that scores the board state.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "CNN-based AI that scores the board state."
        
    def __init__(self):
        Engine.__init__(self)
        nn.Module.__init__(self)

        self.feature_extractor = ChessEmbedding()
        self.score_classifier = ScoreClassifier()
        """return [B, W] where B is the proba of Black winning and W is the proba of White winning"""

        self.legal_move_classifier = LegalMoveClassifier()
        """return [64, 64] for the move matrix and [5] for the promotion choices"""

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.to(self.device)

        self.contrastive_learning = ContrastiveLearning(self.device)

        self.game = Game() # Access to the game toolkit for torch

        self.embedding_loaded = False
        self.classifier_loaded = False

    def play(self) -> chess.Move:
        possible_moves = list(self.game.board.legal_moves)
        move_scores = {}

        if not self.embedding_loaded:
            self.load("models/cnn/cnn_score_embedding.pth", "feature-extractor")
            self.embedding_loaded = True
        if not self.classifier_loaded:
            self.load("models/cnn/cnn_score_classifier.pth", "score-classifier")
            self.classifier_loaded = True
        
        with torch.no_grad():
            for move in possible_moves:
                self.game.board.push(move)
                one_hot = torch.tensor(self.game.one_hot(), dtype=torch.float32).to(self.device)
                turn = torch.tensor(int(self.game.board.turn), dtype=torch.float32).to(self.device)

                score = self.predict(one_hot, turn).cpu().numpy()[0] # ignore the x_t

                if self.color == chess.WHITE:
                    score = score[0]
                else:
                    score = score[1]

                move_scores[move] = score
                
                self.game.board.pop()
        
        for m, s in move_scores.items():
            print(f"{m} -> {s}")
        best_move = max(move_scores, key=move_scores.get)
        return best_move

    def fit(self, one_hots, turns, y, epochs=10, batch_size=32):
        """
        Fit the model to the data.
        
        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param y: list of labels (0 if White wins, 1 if Black wins)
        :param epochs: number of epochs
        :param batch_size: size of the batches
        """

        # Apply contrastive transformation
        X, X_t = self.contrastive_learning.transform(one_hots, turns)
        y = torch.tensor(y, dtype=torch.long, device=self.device)  # Labels for classification
        class_counts = torch.bincount(y)
        class_weights = class_counts.float().sum() / class_counts.float()

        dataset = torch.utils.data.TensorDataset(X, X_t, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(y.device))
        contrastive_loss = nn.CosineEmbeddingLoss()

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, X_t_batch, y_batch in dataloader:
                optimizer.zero_grad()

                # Compute embeddings
                latent = self.feature_extractor(X_batch)
                latent_t = self.feature_extractor(X_t_batch)

                # Flatten embeddings for classification
                latent = latent.view(latent.shape[0], -1)
                latent_t = latent_t.view(latent_t.shape[0], -1)

                # Classification loss
                y_pred = self.score_classifier(latent)
                y_pred_t = self.score_classifier(latent_t)

                class_loss = criterion(y_pred, y_batch)

                # Contrastive loss (forcing X and X_t to have similar embeddings)
                target = torch.ones(latent.shape[0], device=latent.device)
                cont_loss = contrastive_loss(latent, latent_t, target)

                # Total loss
                loss = class_loss + 0.1 * cont_loss + 0.1 * (y_pred - y_pred_t).pow(2).mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader)}")

    def fit_legal_moves(self, one_hots, turns, moves, epochs=10, batch_size=32):
        """
        Train the legal move classifier using contrastive learning.

        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param moves: list of lists of move strings (e.g., ["e2e4", "g1f3", "e7e8q"]) per board
        :param epochs: number of training epochs
        :param batch_size: training batch size
        """

        # Prepare move labels
        move_labels = []
        promo_labels = []

        for move_list in moves:
            move_matrix = torch.zeros((64, 64), dtype=torch.float32)
            promo_vector = torch.zeros(5, dtype=torch.float32)

            for move in move_list:
                start_square = self.game.get_box_idx(move[:2])
                end_square = self.game.get_box_idx(move[2:4])

                move_matrix[start_square, end_square] = 1.0

                if len(move) > 4:  # Promotion case
                    promo_map = {'q': 0, 'r': 1, 'b': 2, 'n': 3}
                    if move[4] in promo_map:
                        promo_vector[promo_map[move[4]]] = 1.0

            move_labels.append(move_matrix)
            promo_labels.append(promo_vector)

        # Convert to tensors
        move_labels = torch.stack(move_labels).to(self.device)
        promo_labels = torch.stack(promo_labels).to(self.device)

        # Apply contrastive transformation
        X, X_t = self.contrastive_learning.transform(one_hots, turns)

        dataset = torch.utils.data.TensorDataset(X, X_t, move_labels, promo_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.legal_move_classifier.parameters(), lr=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        contrastive_loss = nn.CosineEmbeddingLoss()

        for epoch in range(epochs):
            total_loss = 0
            for X_batch, X_t_batch, move_y_batch, promo_y_batch in dataloader:
                optimizer.zero_grad()

                # Compute embeddings
                latent = self.feature_extractor(X_batch)
                latent_t = self.feature_extractor(X_t_batch)

                # Flatten embeddings
                latent = latent.view(latent.shape[0], -1)
                latent_t = latent_t.view(latent_t.shape[0], -1)

                # Predict legal moves
                move_pred, promo_pred = self.legal_move_classifier(latent)
                move_pred_t, promo_pred_t = self.legal_move_classifier(latent_t)

                # Compute losses
                move_loss = criterion(move_pred, move_y_batch)
                promo_loss = criterion(promo_pred, promo_y_batch)

                # Contrastive loss to encourage similar embeddings
                target = torch.ones(latent.shape[0], device=latent.device)
                cont_loss = contrastive_loss(latent, latent_t, target)

                # Consistency loss between transformed predictions
                consistency_loss = (move_pred - move_pred_t).pow(2).mean() + (promo_pred - promo_pred_t).pow(2).mean()

                # Total loss
                loss = move_loss + 0.2 * promo_loss + 0.2 * cont_loss + 0.1 * consistency_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"[{epoch+1}/{epochs}] Loss: {total_loss/len(dataloader)}")

    def evaluate(self, one_hots, turns, y):
        """
        Evaluate the model's accuracy on the given dataset.

        :param one_hots: list of one-hot encoded board states
        :param turns: list of turns (0 for White, 1 for Black)
        :param y: list of labels (0 if White wins, 1 if Black wins)
        :return: accuracy of the model
        """
        self.eval()  # Set the model to evaluation mode
        correct = 0
        total = 0

        y = torch.tensor(y, dtype=torch.long, device=self.device)  # Convert labels to tensor

        with torch.no_grad():
            for i in range(len(one_hots)):
                # Get model prediction
                y_pred = self.predict(one_hots[i], turns[i])
                predicted_label = torch.argmax(y_pred).item()  # Get predicted class (0 or 1)

                if predicted_label == y[i].item():
                    correct += 1
                total += 1

                if i % 100 == 0:
                    print(f"|___[{i}/{len(one_hots)}] Accuracy: {correct / total:.2%}")


        return correct / total if total > 0 else 0

    def predict(self, one_hot, turn):

        one_hot = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)
        turn = torch.tensor(turn, dtype=torch.float32).unsqueeze(0)

        X, X_t = self.contrastive_learning.transform(one_hot, turn)

        with torch.no_grad():
            latent = self.feature_extractor(X)
            latent_t = self.feature_extractor(X_t)

            latent = latent.view(1, -1)
            latent_t = latent_t.view(1, -1)

            # concat latent for batch norm (need at least 2 batch)
            latent = torch.cat((latent, latent_t), dim=0)

            y_pred = self.score_classifier(latent)

            # remove batch dimension
            y_pred = y_pred.squeeze(0)

            return y_pred # [B, W] where B is the proba of Black winning and W is the proba of White winning
        
    def fit_puzzles(self, one_hots, turns, best_moves, epochs=10, batch_size=32):
        """
        Train the model to prioritize the best move.

        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param best_moves: list of best moves in UCI format
        :param epochs: number of epochs
        :param batch_size: batch size
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
        margin_ranking_loss = nn.MarginRankingLoss(margin=0.5)  # Encourages best move to rank higher
        mse_loss = nn.MSELoss()  # Stability loss to regularize probabilities
        contrastive_loss = nn.CosineEmbeddingLoss()

        dataset = list(zip(one_hots, turns, best_moves))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            total_loss = 0

            for one_hot, turn, best_move in dataloader:
                optimizer.zero_grad()

                # Convert inputs to tensors
                one_hot = one_hot.to(self.device)
                turn = turn.to(self.device)

                # Contrastive augmentation
                X, X_t = self.contrastive_learning.transform(one_hot, turn)

                # Extract latent representations
                latent = self.feature_extractor(X)
                latent_t = self.feature_extractor(X_t)

                # Flatten embeddings
                latent = latent.view(latent.shape[0], -1)
                latent_t = latent_t.view(latent_t.shape[0], -1)

                # Compute contrastive loss (forcing embeddings to be similar)
                contrast_loss = contrastive_loss(latent, latent_t, torch.ones(latent.shape[0], device=self.device))

                # Move probability predictions
                move_probs = self.score_classifier(latent)  # Shape: (batch, 64, 64)
                print(move_probs.shape)
                for idx in range(len(best_move)):

                    # Get indices for the best move
                    best_from = self.game.get_box_idx(best_move[idx][:2])  # First 2 chars -> index
                    best_to = self.game.get_box_idx(best_move[idx][2:4])    # Last 2 chars -> index

                    # Extract probabilities for best moves
                    best_move_scores = move_probs[idx], best_from, best_to  # Shape: (idx,)

                    # Sample alternative legal moves
                    random_from = torch.randint(0, 64, (batch_size,), device=self.device)
                    random_to = torch.randint(0, 64, (batch_size,), device=self.device)
                    random_scores = move_probs[:, random_from, random_to]

                    # Ranking target (best move should be ranked higher than alternatives)
                    ranking_target = torch.ones_like(best_move_scores, device=self.device)

                    # Compute ranking loss (best moves > random moves)
                    ranking_loss = margin_ranking_loss(best_move_scores, random_scores, ranking_target)

                    # Regularization: Encourage best move to be close to 1, others to 0
                    reg_loss = mse_loss(best_move_scores, torch.ones_like(best_move_scores, device=self.device))

                    # Total loss (encourage best move, discourage random ones, maintain embedding stability)
                    loss = ranking_loss + 0.1 * reg_loss + 0.1 * contrast_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

            print(f"[{epoch+1}/{epochs}] Loss: {total_loss / len(dataloader)}")

    def evaluate_legal_moves(self, one_hots, turns, y):
        """
        Evaluate the legal move classifier by comparing predicted legal moves
        against the actual legal moves, considering both false positives and false negatives.

        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param y: list of lists containing the legal moves in UCI format for each board
        :return: accuracy of the legal move classifier
        """
        TP = 0  # True Positives
        FP = 0  # False Positives
        FN = 0  # False Negatives

        with torch.no_grad():
            for i, (one_hot, turn, legal_moves) in enumerate(zip(one_hots, turns, y)):
                # Convert board state to tensor
                one_hot = torch.tensor(one_hot, dtype=torch.float32, device=self.device).unsqueeze(0)
                turn = torch.tensor(turn, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Contrastive transformation and feature extraction
                X, X_t = self.contrastive_learning.transform(one_hot, turn)
                latent = self.feature_extractor(X)
                latent_t = self.feature_extractor(X_t)

                latent = latent.view(1, -1)
                latent_t = latent_t.view(1, -1)

                # Concatenate for batch norm
                latent = torch.cat((latent, latent_t), dim=0)

                # Get predictions (updated to use tuple return)
                move_probs, promo_probs = self.legal_move_classifier(latent)

                move_matrix_pred = move_probs[0].cpu().numpy()  # Directly take batch index 0
                promotions_pred = promo_probs[0].cpu().numpy()  # (5) promotion choices

                # Convert predictions to numpy
                # move_matrix_pred = move_probs.squeeze(0).cpu().numpy().reshape(64, 64)  # (64, 64) move matrix
                # promotions_pred = promo_probs.squeeze(0).cpu().numpy()  # (5) promotion choices

                # Convert ground-truth legal moves to indices
                legal_indices = set()
                for move in legal_moves:
                    from_sq = self.game.get_box_idx(move[:2])  # Extract "e2"
                    to_sq = self.game.get_box_idx(move[2:4])  # Extract "e4"
                    promotion = move[4:]  # Extract "q" if present, else empty

                    promo_idx = ["q", "r", "b", "n", ""].index(promotion)
                    legal_indices.add((from_sq, to_sq, promo_idx))

                # Compare predictions with ground truth
                predicted_indices = set()
                for from_sq in range(64):
                    for to_sq in range(64):
                        if move_matrix_pred[from_sq, to_sq] > 0.5:  # Move threshold
                            for promo_idx in range(5):  # Promotion moves
                                if promotions_pred[promo_idx] > 0.5:
                                    predicted_indices.add((from_sq, to_sq, promo_idx))
                                elif promo_idx == 4:  # Non-promotion move (empty string)
                                    predicted_indices.add((from_sq, to_sq, promo_idx))

                # Compute TP, FP, FN
                TP += len(predicted_indices & legal_indices)  # Correctly predicted moves
                FP += len(predicted_indices - legal_indices)  # False positives (illegal moves predicted)
                FN += len(legal_indices - predicted_indices)  # False negatives (missed legal moves)

                if i % 100 == 0:
                    print(f"|___[{i}/{len(one_hots)}] Accuracy: {TP / (TP + FP + FN):.2%}")

        # Compute accuracy (Avoid division by zero)
        total = TP + FP + FN
        return TP / total if total > 0 else 0

    def evaluate_puzzles(self, one_hots, turns, best_moves):
        """
        Evaluate the model by checking how well it ranks the best move higher than others.
        
        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param best_moves: list of best moves in UCI format
        :return: ranking accuracy (how often best move is ranked highest)
        """
        correct_top1 = 0  # How often the best move is the highest-ranked
        total = 0
        mean_rank = 0

        with torch.no_grad():
            for one_hot, turn, best_move in zip(one_hots, turns, best_moves):
                # Convert input to tensor

                X, X_t = self.contrastive_learning.transform(one_hot.unsqueeze(0), turn.unsqueeze(0))

                # Extract latent representation
                latent = self.feature_extractor(X)
                latent_t = self.feature_extractor(X_t)

                latent = latent.view(1, -1)
                latent_t = latent_t.view(1, -1)

                # Concatenate for batch norm
                latent = torch.cat((latent, latent_t), dim=0)

                # Get move probabilities
                move_probs, _ = self.score_classifier(latent)
                move_probs = move_probs[0].squeeze(0).cpu().numpy()  # Convert to NumPy

                # Convert best move to indices
                best_from = self.game.get_box_idx(best_move[:2])
                best_to = self.game.get_box_idx(best_move[2:4])
                best_promo = ["q", "r", "b", "n", ""].index(best_move[4] if len(best_move) > 4 else "")

                # Score of the best move
                best_move_score = move_probs[best_from, best_to]

                # Flatten move matrix and get ranked indices
                move_scores = move_probs.flatten()
                sorted_indices = np.argsort(move_scores)[::-1]  # Sort in descending order

                # Compute rank of the best move
                best_move_rank = np.where(sorted_indices == (best_from * 64 + best_to))[0][0] + 1  # 1-based rank

                # Check if the best move was ranked #1
                if best_move_rank == 1:
                    correct_top1 += 1

                # Track mean rank
                mean_rank += best_move_rank
                total += 1

        # Compute metrics
        top1_accuracy = correct_top1 / total if total > 0 else 0
        avg_rank = mean_rank / total if total > 0 else 0

        # print(f"Top-1 Accuracy: {top1_accuracy:.2%}")
        # print(f"Mean Rank of Best Move: {avg_rank:.2f}")

        return {"top1_accuracy": top1_accuracy, "mean_rank": avg_rank}


    def save(self, path, element):
        """
        Save the model weights to a file.
        element: str, must be 'score-classifier' or 'feature-extractor' or 'legal-move-classifier'
        """
        if element == "score-classifier":
            torch.save(self.score_classifier.state_dict(), path)
        elif element == "feature-extractor":
            torch.save(self.feature_extractor.state_dict(), path)
        elif element == "legal-move-classifier":
            torch.save(self.legal_move_classifier.state_dict(), path)
        else:
            raise ValueError("Invalid element, must be 'score-classifier' or 'feature-extractor' or 'legal-move-classifier'")
        
    def load(self, path, element):
        """
        Load the model weights from a file.
        element: str, must be 'score-classifier' or 'feature-extractor' or 'legal-move-classifier'
        """

        if element == "score-classifier":
            self.score_classifier.load_state_dict(torch.load(path, weights_only=True))
        elif element == "feature-extractor":
            self.feature_extractor.load_state_dict(torch.load(path, weights_only=True))
        elif element == "legal-move-classifier":
            self.legal_move_classifier.load_state_dict(torch.load(path, weights_only=True))
        else:
            raise ValueError("Invalid element, must be 'score-classifier' or 'feature-extractor' or 'legal-move-classifier'")


class ScoreClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.MLP1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),  # ✅ Better than BatchNorm for small batches
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # No softmax (loss handles it)
        )

    def forward(self, x):
        x = self.MLP1(x)  # Shape: (batch, 256)
        x = self.MLP2(x)  # Shape: (batch, 2)

        return F.log_softmax(x, dim=1)  # ✅ Works with NLLLoss
    
class LegalMoveClassifier(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2)
        )

        # Instead of separate "from" and "to" squares, we predict a 64x64 move mask
        self.move_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64)  # Predicts move matrix
        ) #nn.Linear(128, 64 * 64)  # Predicts move matrix
        self.promotion_head = nn.Linear(128, 5)  # Predicts promotion choices

    def forward(self, x):

        x = x.unsqueeze(1)  # Add sequence dimension (batch, 1, 512)
        x, _ = self.attention(x, x, x)  # Self-attention
        x = x.squeeze(1)  # Remove sequence dimension

        x = self.mlp(x)

        move_logits = self.move_head(x)  # (batch, 64*64)
        move_logits = move_logits.view(-1, 64, 64)  # Reshape to (batch, 64, 64)
        move_probs = torch.sigmoid(move_logits)  # Sigmoid activation for multi-label

        promo_logits = self.promotion_head(x)  # (batch, 5)
        promo_probs = torch.sigmoid(promo_logits)  # Sigmoid for multiple promotions

        return move_probs, promo_probs

class ChessEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.resn1 = ResBlock(14, 64, 128, 3, 1, 1, attention=False)  # (batch, 128, 8, 8)
        self.resn2 = ResBlock(128, 128, 256, 6, 1, 2)  # (batch, 256, 8, 8)
        self.resn3 = ResBlock(256, 256, 512, 3, 2, 1)  # (batch, 512, 4, 4)
        self.resn4 = ResBlock(512, 512, 512, 3, 1, 1)  # (batch, 512, 4, 4)
        self.resn5 = ResBlock(512, 512, 256, 6, 2, 1)  # Changed 512 - 4 * 8 * 8 to 256

        self.heatmap1 = HeatMap()
        self.heatmap2 = HeatMap()
        self.heatmap3 = HeatMap()
        self.heatmap4 = HeatMap()  # (batch, 1, 8, 8)

        # Positional embedding for the heatmap
        self.pos_embed = PositionalEncoding(4)
        self.heatmap_attention = CBAMChannelAttention(4, 4)

        self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Compute heatmaps
        heatmap1 = self.heatmap1(x)
        heatmap2 = self.heatmap2(x)
        heatmap3 = self.heatmap3(x)
        heatmap4 = self.heatmap4(x)

        # Pass through residual blocks
        x = self.resn1(x)
        x = self.dropout(x)
        x = self.resn2(x)
        x = self.dropout(x) #+ heatmap1  # Residual heatmap addition
        x = self.resn3(x)
        x = self.dropout(x)
        x = self.resn4(x)
        x = self.dropout(x)
        x = self.resn5(x)
        x = self.dropout(x) # (batch, 256, 1, 1)

        # Concatenate heatmaps and apply positional encoding + attention
        concat_heatmap = torch.cat((heatmap1, heatmap2, heatmap3, heatmap4), dim=1)  # (batch, 4, 8, 8)
        concat_heatmap = self.pos_embed(concat_heatmap)  # Add positional encoding
        concat_heatmap = self.heatmap_attention(concat_heatmap)  # (batch, 4, 8, 8)

        # Flatten heatmap: (batch, 4, 8, 8) -> (batch, 4 * 8 * 8)
        concat_heatmap = concat_heatmap.view(concat_heatmap.size(0), -1)

        # Flatten CNN output: (batch, 256, 1, 1) -> (batch, 256)
        x = x.view(x.size(0), -1)

        # Concatenate CNN output with heatmap features
        x = torch.cat((x, concat_heatmap), dim=1)  # Adjusted for correct feature size (batchsize, 256 + 256)
        
        # Apply self-attention
        x = x.unsqueeze(1)
        x, _ = self.self_attention(x, x, x)
        x = x.squeeze(1)

        return x

class ContrastiveLearning:

    def __init__(self, device):
        self.device = device

    def transform(self, one_hot, turn):
        """
        Apply transformation:
        - Rotate board 180°.
        - Swap piece colors.
        - Flip turn.
        """
        batch_size = one_hot.shape[0]

        # Rotate board 180 degrees
        one_hot_rotated = torch.rot90(one_hot, 2, (1, 2))  # (batch, 8, 8, 13)

        # Define piece swap mapping (K↔k, Q↔q, etc.)
        # [K, Q, R, B, N, P, k, q, r, b, n, p, EMPTY] -> [k, q, r, b, n, p, K, Q, R, B, N, P, EMPTY]
        swap_indices = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12]
        one_hot_transformed = one_hot_rotated[..., swap_indices]  # Apply swap

        # Flip turn
        turn_transformed = 1 - turn  # 0 ↔ 1

        # Expand turn to [batch_size, 8, 8, 1] and concatenate
        turn_expanded = turn.view(batch_size, 1, 1, 1).expand(-1, 8, 8, 1)
        x = torch.cat((one_hot, turn_expanded), dim=-1)  # [batch_size, 8, 8, 14]

        turn_expanded_t = turn_transformed.view(batch_size, 1, 1, 1).expand(-1, 8, 8, 1)
        x_t = torch.cat((one_hot_transformed, turn_expanded_t), dim=-1)

        x = x.permute(0, 3, 2, 1).to(self.device)
        x_t = x_t.permute(0, 3, 2, 1).to(self.device)

        return x, x_t
    
class HeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(14, 64, 3, 1, 1)  # Output: (batch, 64, 8, 8)
        # We'll manually pad before conv2 to simulate "same" padding for a 6x6 kernel.
        # For an input of size 8, kernel=6, stride=1, the total padding needed is k - 1 = 5.
        # We'll pad left=2, right=3, top=2, bottom=3.
        self.pad = nn.ZeroPad2d((2, 3, 2, 3))
        self.conv2 = nn.Conv2d(64, 64, 6, 1, 0)  # No automatic padding here
        self.conv3 = nn.Conv2d(64, 32, 5, 1, 2)  # Output: (batch, 32, 8, 8) with kernel=5, padding=2
        self.conv4 = nn.Conv2d(32, 1, 3, 1, 1)   # Output: (batch, 1, 8, 8)
        
        self.res_connection = nn.Conv2d(14, 64, 3, 1, 1)  # Ensures (batch, 64, 8, 8)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = self.res_connection(x)         # (batch, 64, 8, 8)
        x = self.conv1(x)                           # (batch, 64, 8, 8)
        x = self.relu(x)
        x = self.pad(x)                             # Manual padding: input 8 becomes 8+2+3 = 13
        x = self.conv2(x)                           # (batch, 64, 13-6+1=8, 8) => (batch, 64, 8, 8)
        x += residual                               # Shapes match: (batch, 64, 8, 8)
        x = self.relu(x)
        x = self.conv3(x)                           # (batch, 32, 8, 8)
        x = self.relu(x)
        x = self.conv4(x)                           # (batch, 1, 8, 8)
        x = self.sigmoid(x)
        return x