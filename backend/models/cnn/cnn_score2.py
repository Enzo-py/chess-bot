import numpy as np
from models.deep_engine import DeepEngine
from models.cnn.cnn_toolbox import MultiHeadCrossAttention, CBAM, DepthwiseResBlock, RelativePositionalEncoding, SEAttention

from src.chess.game import Game

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNScoreTmp(DeepEngine):
    """
    CNN-based AI that scores the board state.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "CNN-based AI that scores the board state."
        
    def __init__(self):
        super().__init__()

        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessEmbedding2())
        self.set(head_name="decoder", head=Decoder())

    
class GenerativeHead(nn.Module):
    def __init__(self):
        super().__init__()

        # MLP Feature Expansion
        self.fc1 = nn.Linear(512 + 256, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.shortcut1 = nn.Linear(512 + 256, 2048)

        self.fc1_2 = nn.Linear(512 + 256, 1024)
        self.fc2_2 = nn.Linear(1024, 2048)
        self.shortcut1_2 = nn.Linear(512 + 256, 2048)

        self.gate_fc = nn.Linear(2048 * 2, 2048)
        self.cross_attention = MultiHeadCrossAttention(2048, 4)

        self.projection = nn.Linear(2048, 64*64*5)

        # Normalization and Dropout
        self.norm1_1 = nn.LayerNorm(1024)
        self.norm2_1 = nn.LayerNorm(2048)
        self.norm1_2 = nn.LayerNorm(1024)
        self.norm2_2 = nn.LayerNorm(2048)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # MLP Expansion with Residual Connections
        shortcut1 = self.shortcut1(x)
        shortcut1_2 = self.shortcut1_2(x)

        x_1 = F.relu(self.fc1(x))
        x_1 = self.norm1_1(x_1)
        x_1 = self.dropout(x_1)
        x_1 = F.relu(self.fc2(x_1))
        x_1 = self.norm2_1(x_1)
        x_1 += shortcut1

        x_2 = F.relu(self.fc1_2(x))
        x_2 = self.norm1_2(x_2)
        x_2 = self.dropout(x_2)
        x_2 = F.relu(self.fc2_2(x_2))
        x_2 = self.norm2_2(x_2)
        x_2 += shortcut1_2

        x_cat = torch.cat([x_1, x_2], dim=-1)  # shape: (batch, 4096)
        gate = torch.sigmoid(self.gate_fc(x_cat))  # shape: (batch, 2048)

        # Fuse the two branches using the learned gate:
        # gate * x_1 + (1 - gate) * x_2
        x_fused = gate * x_1 + (1 - gate) * x_2

        # Compute cross attention (here using x₁ as query and x₂ as key/value)
        x_cross = self.cross_attention(x_1, x_2)

        # Combine the cross attention output with the gated fusion.
        # This lets the model leverage both learned interactions and a direct fusion.
        x_combined = x_cross + x_fused

        # Projection
        x = self.projection(x_combined)
        return x

class BoardEvaluator(nn.Module):
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
        x = self.MLP1(x)  # (batch, 256)
        return F.softmax(x, dim=1)

class HeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
        self.se = SEAttention(64)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.se(x)  # Apply attention
        x = self.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x  # (batch, 1, 8, 8)

class ChessEmbedding2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_encoding = RelativePositionalEncoding(13) # avoid equivariance to translation

        self.res1 = DepthwiseResBlock(13, 128, 3, 1, 1)
        self.res2 = DepthwiseResBlock(128, 256, 3, 1, 1)
        self.res3 = DepthwiseResBlock(256, 512, 3, 2, 1)
        self.res4 = DepthwiseResBlock(512, 512, 3, 2, 1)
        self.res5 = DepthwiseResBlock(512, 1024, 3, 2, 1)

        # # CBAM Attention
        # self.cbam1 = CBAM(128)
        # self.cbam2 = CBAM(256)
        # self.cbam3 = CBAM(512)
        # self.cbam4 = CBAM(512)
        # self.cbam5 = CBAM(1024)

        # Heatmaps
        self.heatmap1 = HeatMap()
        self.heatmap2 = HeatMap()
        self.heatmap3 = HeatMap()
        self.heatmap4 = HeatMap()

        self.proj = nn.Linear(1024 + 256, 1024)
        self.norm = nn.LayerNorm(1024)
        self.fc = nn.Linear(1024, 512 + 256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.pos_encoding(x)

        heatmap1 = self.heatmap1(x)
        heatmap2 = self.heatmap2(x)
        heatmap3 = self.heatmap3(x)
        heatmap4 = self.heatmap4(x)

        # CNN + CBAM Attention
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.dropout(x)
        x = self.res4(x)
        x = self.res5(x)

        x = x.view(x.shape[0], -1)
        heatmaps = torch.cat([heatmap1, heatmap2, heatmap3, heatmap4], dim=1)
        heatmaps = heatmaps.view(heatmaps.shape[0], -1)

        x = torch.cat([x, heatmaps], dim=1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
class Decoder(nn.Module):
    """from latent to board"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc1 = nn.Linear(512 + 256, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc_board_1 = nn.Linear(1024, 1024)
        self.fc_board_2 = nn.Linear(1024, 8*8*12)
        
        self.fc_turn = nn.Linear(1024, 1)

    def forward(self, x):
        """
        x: [batch_size, 256]
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
