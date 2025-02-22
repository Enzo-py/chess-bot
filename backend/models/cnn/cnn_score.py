import numpy as np
from models.deep_engine import DeepEngine, ScoreModel
from models.cnn.cnn_toolbox import ResBlock, PositionalEncoding, CBAMChannelAttention

from src.chess.game import Game

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNScore(DeepEngine):
    """
    CNN-based AI that scores the board state.
    """

    __author__ = "Enzo Pinchon"
    __description__ = "CNN-based AI that scores the board state."
        
    def __init__(self):
        super().__init__()

        self.score_function = ScoreModelCNN()

    
class ScoreModelCNN(ScoreModel):
    def __init__(self):
        super().__init__()

        self.embedding = ChessEmbedding()
        self.classifier = ScoreClassifier()
        self.generative_head = GenerativeHead()

        self.set_device()

class GenerativeHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(512+256, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.fc3 = nn.Linear(2048, 64*64*5)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ScoreClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.MLP1 = nn.Sequential(
            nn.Linear(512+256, 256),
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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.MLP1(x)  # Shape: (batch, 256)
        x = self.MLP2(x)  # Shape: (batch, 2)

        return F.softmax(x, dim=1)
    
class ChessEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.resn1 = ResBlock(14, 64, 128, 3, 1, 1, attention=False)  # (batch, 128, 8, 8)
        self.resn2 = ResBlock(128, 128, 256, 6, 1, 2)  # (batch, 256, 8, 8)
        self.resn3 = ResBlock(256, 256, 512, 3, 2, 1)  # (batch, 512, 4, 4)
        self.resn4 = ResBlock(512, 512, 512, 3, 1, 1)  # (batch, 512, 4, 4)
        self.resn5 = ResBlock(512, 512, 512, 6, 2, 1)  # Changed 512 - 4 * 8 * 8 to 256

        self.heatmap1 = HeatMap()
        self.heatmap2 = HeatMap()
        self.heatmap3 = HeatMap()
        self.heatmap4 = HeatMap()  # (batch, 1, 8, 8)

        # Positional embedding for the heatmap
        self.pos_embed = PositionalEncoding(4)
        self.heatmap_attention = CBAMChannelAttention(4, 4)

        # self.self_attention = nn.MultiheadAttention(embed_dim=512, num_heads=4, batch_first=True)

        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, 8, 8, 14) -> (batch, 14, 8, 8)
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

        # # Flatten heatmap: (batch, 4, 8, 8) -> (batch, 4 * 8 * 8)
        concat_heatmap = concat_heatmap.view(concat_heatmap.size(0), -1)

        # Flatten CNN output: (batch, 256, 1, 1) -> (batch, 256)
        x = x.view(x.size(0), -1)

        # # Concatenate CNN output with heatmap features
        x = torch.cat((x, concat_heatmap), dim=1)  # Adjusted for correct feature size (batchsize, 512 + 256)
        
        # # Apply self-attention
        # x = x.unsqueeze(1)
        # x, _ = self.self_attention(x, x, x)
        # x = x.squeeze(1)

        return x

   
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