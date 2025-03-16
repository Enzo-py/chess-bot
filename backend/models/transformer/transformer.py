import numpy as np
from models.deep_engine import DeepEngine
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.transformer.transformer_toolbox import (
    PositionalEmbedding,
    SelfAttention,
    FeedForward,
    TransformerBlock,
    PieceEmbedding,
    ChessCrossAttention
)

from src.chess.game import Game
import chess

class TransformerScore(DeepEngine):
    """
    Transformer-based AI that scores the board state.
    """

    __author__ = "AI Assistant"
    __description__ = "Transformer-based AI that scores the board state."
        
    def __init__(self):
        super().__init__()

        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessTransformerEncoder())
        self.set(head_name="decoder", head=Decoder())


class ChessTransformerEncoder(nn.Module):
    """
    Transformer-based encoder that processes the chess board state.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=4, ff_dim=512, dropout=0.1):
        super().__init__()
        
        # Piece embedding layer
        self.piece_embedding = PieceEmbedding(embed_dim)
        
        # Positional embedding
        self.pos_embedding = PositionalEmbedding(embed_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Initial convolution to extract local patterns (optional)
        self.conv_embed = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Projection to connect conv features with transformer
        self.conv_projection = nn.Linear(128, embed_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Final output projection
        self.output_projection = nn.Linear(embed_dim * 64, 512 + 256)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x comes in as [batch, 8, 8, 13]
        batch_size = x.size(0)
        
        # Convolutional feature extraction path
        conv_x = x.permute(0, 3, 1, 2)  # [batch, 13, 8, 8]
        conv_x = self.conv_embed(conv_x)  # [batch, 128, 8, 8]
        conv_x = conv_x.permute(0, 2, 3, 1)  # [batch, 8, 8, 128]
        conv_x = conv_x.reshape(batch_size, 64, 128)  # [batch, 64, 128]
        conv_features = self.conv_projection(conv_x)  # [batch, 64, embed_dim]
        
        # Transformer path with embeddings
        # Get piece embeddings
        piece_embeddings = self.piece_embedding(x)  # [batch, 64, embed_dim]
        
        # Combine features and add positional embeddings
        transformer_input = piece_embeddings + conv_features  # [batch, 64, embed_dim]
        transformer_input = self.pos_embedding(transformer_input)  # Add positional info
        transformer_input = self.dropout(transformer_input)
        
        # Pass through transformer blocks
        x = transformer_input
        for block in self.transformer_blocks:
            x = block(x)
        
        # Final layer normalization
        x = self.norm(x)
        
        # Flatten to create a global representation
        x = x.reshape(batch_size, -1)  # [batch, 64*embed_dim]
        
        # Project to the expected output dimension
        x = self.output_projection(x)  # [batch, 512+256]
        
        return x


class BoardEvaluator(nn.Module):
    """
    Evaluates the board position to determine win probabilities.
    """
    def __init__(self):
        super().__init__()

        # MLP for feature vector
        self.MLP1 = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.MLP1(x)
        return F.softmax(x, dim=1)


class GenerativeHead(nn.Module):
    """
    Generates move probabilities for each possible move.
    """
    def __init__(self):
        super().__init__()
        
        # Dimensions
        self.latent_dim = 512 + 256
        self.hidden_dim = 1024
        self.final_dim = 2048
        
        # MLP feature expansion
        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.final_dim)
        self.shortcut1 = nn.Linear(self.latent_dim, self.final_dim)
        
        # Cross-attention for move generation
        self.cross_attention = ChessCrossAttention(self.final_dim, num_heads=8)
        
        # Final projection to move space
        self.projection = nn.Linear(self.final_dim, 64*64*5)
        
        # Normalization and dropout
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.final_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # First path
        shortcut = self.shortcut1(x)
        
        # Main path with MLP
        main_path = F.relu(self.fc1(x))
        main_path = self.norm1(main_path)
        main_path = self.dropout(main_path)
        main_path = F.relu(self.fc2(main_path))
        main_path = self.norm2(main_path)
        
        # Add residual connection
        x = main_path + shortcut
        
        # Reshape for attention
        batch_size = x.size(0)
        query = x.unsqueeze(1)  # [batch, 1, final_dim]
        key_value = query.clone()
        
        # Apply self-attention
        x = self.cross_attention(query, key_value).squeeze(1)
        
        # Project to move space
        x = self.projection(x)
        
        return x


class Decoder(nn.Module):
    """
    Decodes latent representation back to a board state.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(512 + 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_board_1 = nn.Linear(1024, 1024)
        self.fc_board_2 = nn.Linear(1024, 8*8*12)
        
        self.fc_turn = nn.Linear(1024, 1)

    def forward(self, x):
        """
        x: [batch_size, 768]
        """
        if isinstance(x, tuple):
            x = x[0]

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_board = F.relu(self.fc_board_1(x))
        x_board = F.relu(self.fc_board_2(x))
        x_board = x_board.view(-1, 8, 8, 12)

        x_turn = self.fc_turn(x)
        return x_board, x_turn 