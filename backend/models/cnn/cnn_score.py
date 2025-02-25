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

        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.shortcut = nn.Linear(512, 2048)  # Residual connection

        self.fc3 = nn.Linear(2048, 8 * 8 * 128)  # Project to spatial features
        self.deconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 8x8 → 16x16
        self.deconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1, dilation=2)  # 16x16 → 32x32
        self.deconv3 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1)  # 32x32 → 64x64
        self.deconv4 = nn.Conv2d(16, 5, 3, padding=1)  # Final 64x64 output (no stride)

        self.norm1 = nn.GroupNorm(16, 1024)  # More stable than BatchNorm
        self.norm2 = nn.GroupNorm(16, 2048)
        self.dropout = nn.Dropout(0.2)  # Increased for stability

    def forward(self, x):
        shortcut = self.shortcut(x)  # Save input for residual connection
        x = F.gelu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)

        x = F.gelu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout(x)

        x = x + shortcut  # Apply residual connection

        x = self.fc3(x)
        x = x.view(-1, 128, 8, 8)  # Reshape for CNN decoding

        x = F.gelu(self.deconv1(x))  # 8x8 → 16x16
        x = F.gelu(self.deconv2(x))  # 16x16 → 32x32
        x = F.gelu(self.deconv3(x))  # 32x32 → 64x64
        x = self.deconv4(x)  # Final output (batch, 5, 64, 64)

        batch_size = x.shape[0]
        return x.view(batch_size, 5, 64 * 64)

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
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.MLP1(x)  # Shape: (batch, 256)
        x = self.MLP2(x)  # Shape: (batch, 2)

        return F.softmax(x, dim=1)
    
class DepthwiseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
    def forward(self, x):
        res = self.residual(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x + res)


# 🔹 SE Attention for Heatmaps
class SEAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class HeatMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(13, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, 5, padding=2)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.norm = nn.BatchNorm2d(64)
        self.act = nn.ReLU()
        self.se = SEAttention(64)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.norm(x)
        x = self.se(x)  # Apply attention
        x = self.act(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x  # (batch, 1, 8, 8)

class ChessEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.res1 = DepthwiseResBlock(13, 128, 3, 1, 1)   # (batch, 128, 8, 8)
        self.res2 = DepthwiseResBlock(128, 256, 3, 1, 1)  # (batch, 256, 8, 8)
        self.res3 = DepthwiseResBlock(256, 512, 3, 2, 1)  # (batch, 512, 4, 4)
        self.res4 = DepthwiseResBlock(512, 512, 5, 2, 2)  # (batch, 512, 2, 2)
        self.res5 = DepthwiseResBlock(512, 512, 3, 2, 1)  # (batch, 512, 1, 1)

        # Heatmaps
        self.heatmap1 = HeatMap()
        self.heatmap2 = HeatMap()
        self.heatmap3 = HeatMap()
        self.heatmap4 = HeatMap()

        # Final embedding projection
        self.proj = nn.Linear(512 + 256, 512)   # Project concatenated output to fixed dim
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (batch, 8, 8, 14) -> (batch, 14, 8, 8)

        # Compute heatmaps
        heatmap1 = self.heatmap1(x)
        heatmap2 = self.heatmap2(x)
        heatmap3 = self.heatmap3(x)
        heatmap4 = self.heatmap4(x)

        # CNN Feature Extraction
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)  # (batch, 512, 1, 1)

        # Flatten outputs
        x = x.view(x.shape[0], -1)  # (batch, 512)
        heatmaps = torch.cat([heatmap1, heatmap2, heatmap3, heatmap4], dim=1)  # (batch, 4, 8, 8)
        heatmaps = heatmaps.view(heatmaps.shape[0], -1)  # Flatten (batch, 4 * 8 * 8)

        # Concatenate and project
        x = torch.cat([x, heatmaps], dim=1)  # (batch, 512 + 4 * 8 * 8)
        x = self.proj(x)
        x = self.norm(x)

        return x  # Final embedding (batch, 512)
