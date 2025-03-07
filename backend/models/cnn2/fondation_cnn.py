import random
from models.engine import Engine

import chess
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FondationCNN(Engine, nn.Module):

    __author__ = "Enzo Pinchon"
    __description__ = "CNN-based AI ..."

    latent_dim = 256
    
    def __init__(self):
        Engine.__init__(self)
        nn.Module.__init__(self)

        self.feature_extractor = FeatureExtraction()

        self.game_state_classifier = nn.Sequential(
            nn.Linear(FondationCNN.latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4) # checkmate, draw, check, normal
        )

        self.next_move_classifier = MoveClassifier(FondationCNN.latent_dim)
        self.move_legality_classifier = MoveLegalityClassifier(FondationCNN.latent_dim)

        self.feature_extractor.load_state_dict(torch.load("./models/cnn/fcnn-Z-move-v1.pth", weights_only=True))
        self.move_legality_classifier.load_state_dict(torch.load("./models/cnn/fcnn-C-move-v1.pth", weights_only=True))
        self.feature_extractor.train()
        self.next_move_classifier.train()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
        self.to(self.device)

    def play(self) -> chess.Move:
        
        one_hot = torch.tensor(self.game.one_hot(), dtype=torch.float32).unsqueeze(0)  # Add batch dim
        turn = torch.tensor([int(self.game.board.turn)], dtype=torch.float32)  # Keep batch dim

        # Move tensors to device
        one_hot, turn = one_hot.to(self.device), turn.to(self.device)

        with torch.no_grad():
            # Extract latent features
            latent = self.feature_extractor(one_hot, turn)  # [1, latent_dim]

            # Predict next move logits
            next_move_logits = self.next_move_classifier(latent)  # [1, 64+64+5]

            # Remove batch dimension
            next_move_logits = next_move_logits.squeeze(0)  # [64+64+5]

            # Split logits into `from`, `to`, and `promotion`
            from_logits, to_logits, promote_logits = next_move_logits[:64], next_move_logits[64:128], next_move_logits[128:]

            # Get the best move indices
            from_idx = torch.argmax(from_logits).item()
            to_idx = torch.argmax(to_logits).item()
            promote_idx = torch.argmax(promote_logits).item()

            # Convert indices to chess move
            promote = ["q", "r", "b", "n", ""][promote_idx]
            from_square = self.game.get_box_label(from_idx).lower()
            to_square = self.game.get_box_label(to_idx).lower()
            move = chess.Move.from_uci(f"{from_square}{to_square}{promote}")

        print(">> Predicted move:", from_square, to_square, promote)
        # Check if move is legal; otherwise, pick a random legal move
        if move in self.game.board.legal_moves:
            return move
        else:
            print(">> Illegal move predicted, playing a random move instead.")
            legal_moves = list(self.game.board.legal_moves)
            return random.choice(legal_moves) if legal_moves else None
            
    def forward(self, one_hot, turn):
        """
            one_hot: [batch_size, 8, 8, 13]
            turn: [batch_size, 1]
        """
        x = self.feature_extractor(one_hot, turn)
        batch_size = x.shape[0]

        # Flatten for fully connected layer
        x = x.view(batch_size, -1)  # [batch_size, 64 * 8 * 8]
        x = self.game_state_classifier(x)  # [batch_size, 4]
        return x


class PositionalEncoding(nn.Module):
    """2D Positional Encoding for Chess Board (8x8)"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Create a fixed positional encoding table
        pe = torch.zeros(8, 8, channels)  # [8,8,C]
        y_pos = torch.arange(0, 8).unsqueeze(1).repeat(1, 8)  # Y-coordinates
        x_pos = torch.arange(0, 8).unsqueeze(0).repeat(8, 1)  # X-coordinates

        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        
        pe[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)

        self.register_buffer("pe", pe)  # Store as a constant tensor

    def forward(self, x):
        """
        x: [batch_size, C, 8, 8]
        """
        return x + self.pe.permute(2, 0, 1).unsqueeze(0)  # Add positional encoding

class FeatureExtraction(nn.Module):
    """
    Feature extraction module: Produces a 256D embedding from an 8x8 chessboard.
    Now includes:
    - Residual connections
    - Strided convolutions for downsampling
    - Positional Encoding
    - Self-Attention for feature interactions
    """

    def __init__(self):
        super().__init__()

        # Positional Encoding (Learnable)
        self.pos_embed = PositionalEncoding(14)  # [1, 14, 8, 8]

        # Initial Convolution + First Residual
        self.conv1 = nn.Conv2d(14, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # ⬇ 8x8 → 4x4
        self.res_conv1 = nn.Conv2d(14, 128, kernel_size=1, stride=2, padding=0)  # Residual downsample

        # More Feature Extraction + Second Residual
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # ⬇ 4x4 → 2x2
        self.res_conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)  # Residual downsample

        # Final Convolutions + Third Residual
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)  # ⬇ 2x2 → 1x1
        self.res_conv3 = nn.Conv2d(256, FondationCNN.latent_dim, kernel_size=1, stride=2, padding=0)  # Residual downsample

        # Self-Attention for Feature Interaction
        self.attn = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True, dropout=0.1)  # 4 attention heads

        # Dropout for Regularization
        self.dropout = nn.Dropout(0.25)

        self.fc = nn.Linear(256, FondationCNN.latent_dim)

    def forward(self, one_hot, turn):
        """
        Args:
            one_hot: [batch_size, 8, 8, 13]
            turn: [batch_size, 1]
        Returns:
            Chessboard embedding: [batch_size, 256]
        """
        batch_size = one_hot.shape[0]

        # Expand turn to [batch_size, 8, 8, 1] and concatenate
        turn_expanded = turn.view(batch_size, 1, 1, 1).expand(-1, 8, 8, 1)
        x = torch.cat((one_hot, turn_expanded), dim=-1)  # [batch_size, 8, 8, 14]

        # Reshape to (batch_size, channels, height, width)
        x = x.permute(0, 3, 1, 2)  # [batch_size, 14, 8, 8]

        # Add Positional Encoding
        x = x + self.pos_embed(x)

        # First Residual Block
        residual1 = self.res_conv1(x)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(x + residual1)  # Residual connection

        # Second Residual Block
        residual2 = self.res_conv2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = F.relu(x + residual2)  # Residual connection

        # Third Residual Block
        residual3 = self.res_conv3(x)
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        x = F.relu(x + residual3)  # Residual connection

        # Self-Attention for Feature Interaction
        x = x.view(batch_size, -1, 256)
        x, _ = self.attn(x, x, x)
        x = x.view(batch_size, -1)

        # Global Pooling to get (batch_size, 256)
        x = x.view(batch_size, -1)  # Flatten to [batch_size, 256]
        x = self.dropout(x)  # Apply dropout before returning

        x = self.fc(x)  # [batch_size, 256]

        return x  # Final embedding (batch_size, 256)

class MoveClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc_from = nn.Linear(128, 64)  # 64 squares for FROM
        self.fc_to = nn.Linear(128, 64)  # 64 squares for TO
        self.fc_promote = nn.Linear(128, 5)  # 5 possible promotions: Q, R, B, N, None

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        from_logits = self.fc_from(x)  # Logits for start position
        to_logits = self.fc_to(x)  # Logits for end position
        promote_logits = self.fc_promote(x)
        logits = torch.cat((from_logits, to_logits, promote_logits), dim=-1)
        return logits

class MoveLegalityClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 64 + 64 + 5, 256)  # 64+64+5: FROM, TO, PROMOTE
        self.fc2 = nn.Linear(256, 128)
        self.res_net = nn.Linear(input_dim, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)  # Binary classification: legal (1) or not (0)

    def forward(self, x, move):
        """
        x: Latent space (batch_size, 256)
        move: Tensor of shape (batch_size, 64+64+5) containing move information
        """
        residual = x
        x = torch.cat((x, move), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x) + self.res_net(residual))
        x = F.relu(self.fc3(x))
        x = torch.softmax(self.fc4(x), dim=-1)
        return x

