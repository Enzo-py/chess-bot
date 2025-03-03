import numpy as np
from models.deep_engine import DeepEngine
from models.cnn.cnn_toolbox import SqueezeExcitation

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

        self.set(head_name="board_evaluation", head=BoardEvaluator())
        self.set(head_name="generative", head=GenerativeHead())
        self.set(head_name="encoder", head=ChessEmbedding())

    
class GenerativeHead(nn.Module):
    def __init__(self):
        super().__init__()

        # Heatmap Feature Extractor (CNN Encoder)
        self.heatmap_encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Reduce spatial size
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Final output (batch, 64, 1, 1)
        )

        # MLP Feature Expansion
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.shortcut1 = nn.Linear(512, 1024)
        self.shortcut2 = nn.Linear(1024, 2048)

        # Heatmap feature fusion
        self.fc_heatmap = nn.Linear(64, 128)
        self.fc_fusion = nn.Linear(2048 + 128, 2048)  # Combine MLP and heatmap features

        # Spatial Projection
        self.fc3 = nn.Linear(2048, 8 * 8 * 128)

        # CNN Decoder with SE Attention
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.se1 = SqueezeExcitation(64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.se2 = SqueezeExcitation(32)

        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1)

        self.final_conv = nn.Conv2d(16, 5, kernel_size=3, padding=1)  # Output: (batch, 5, 64, 64)

        # Normalization and Dropout
        self.norm1 = nn.LayerNorm(1024)
        self.norm2 = nn.LayerNorm(2048)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x, heatmaps = x  # x = (batch, 512), heatmaps = (batch, 4, 8, 8)

        # Process Heatmaps
        heatmap_features = self.heatmap_encoder(heatmaps)  # (batch, 64, 1, 1)
        heatmap_features = heatmap_features.view(x.size(0), -1)  # Flatten to (batch, 64)
        heatmap_features = F.silu(self.fc_heatmap(heatmap_features))  # Expand to (batch, 128)

        # MLP Expansion with Residual Connections
        shortcut1 = self.shortcut1(x)
        x = F.silu(self.fc1(x))
        x = self.norm1(x)
        x = self.dropout(x)
        x = x + shortcut1  # Residual

        shortcut2 = self.shortcut2(x)
        x = F.silu(self.fc2(x))
        x = self.norm2(x)
        x = self.dropout(x)
        x = x + shortcut2  # Residual

        # Fuse Heatmap Features with MLP Features
        x = torch.cat([x, heatmap_features], dim=1)  # (batch, 2048 + 128)
        x = F.silu(self.fc_fusion(x))  # Merge heatmap and MLP representations

        # Spatial Projection
        x = self.fc3(x).view(-1, 128, 8, 8)  # Reshape to (batch, 128, 8, 8)

        # CNN Decoder with Attention
        x = self.upsample1(x)
        x = F.silu(self.conv1(x))
        x = self.se1(x)

        x = self.upsample2(x)
        x = F.silu(self.conv2(x))
        x = self.se2(x)

        x = self.upsample3(x)
        x = F.silu(self.conv3(x))

        x = self.final_conv(x)  # (batch, 5, 64, 64)

        batch_size = x.shape[0]
        return x.view(batch_size, 5 * 64 * 64)

class BoardEvaluator(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional feature extractor for heatmaps
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output shape: (batch, 32, 1, 1)
        )

        # MLP for feature vector
        self.MLP1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(256),
        )

        # Final classifier combining CNN and MLP outputs
        self.MLP2 = nn.Sequential(
            nn.Linear(256 + 32, 128),  # Merge CNN output (32) with MLP output (256)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x, heatmaps = x

        # Process heatmaps using CNN
        heatmap_features = self.cnn(heatmaps)  # (batch, 32, 1, 1)
        heatmap_features = heatmap_features.view(x.size(0), -1)  # Flatten to (batch, 32)

        # Process feature vector
        x = self.MLP1(x)  # (batch, 256)

        # Merge both feature representations
        x = torch.cat([x, heatmap_features], dim=1)  # (batch, 256 + 32)

        # Final classification
        x = self.MLP2(x)  # (batch, 2)

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
        # self.proj = nn.Linear(512, 512)   # Project concatenated output to fixed dim
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
        # heatmaps = heatmaps.view(heatmaps.shape[0], -1)  # Flatten (batch, 4 * 8 * 8)

        # Concatenate and project
        # x = torch.cat([x, heatmaps], dim=1)  # (batch, 512 + 4 * 8 * 8)
        # x = self.proj(x)
        x = self.norm(x)

        return x, heatmaps  # Final embedding (batch, 512), heatmaps 
