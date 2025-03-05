import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_fc1 = nn.Linear(channels, channels // reduction)
        self.channel_fc2 = nn.Linear(channels // reduction, channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.shape
        
        # Channel Attention
        avg_pool = x.mean(dim=(2, 3))
        # Flatten spatial dimensions and then take max along that dimension.
        max_pool, _ = x.view(batch, channels, -1).max(dim=2)
        fc = lambda pool: self.sigmoid(self.channel_fc2(self.relu(self.channel_fc1(pool))))
        channel_attention = (fc(avg_pool) + fc(max_pool)).view(batch, channels, 1, 1)

        # Spatial Attention
        spatial_input = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]], dim=1)
        spatial_attention = self.sigmoid(self.spatial_conv(spatial_input))

        return x * channel_attention * spatial_attention  # Scale feature maps

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

class DepthwiseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_type='batch'):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        if norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'group':
            self.norm = nn.GroupNorm(num_groups=out_channels // 16 if out_channels >= 16 else 1, num_channels=out_channels)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm([out_channels, 1, 1])  # Normalize across channels

        self.act = nn.SiLU()  # Swish activation (better for CNNs)
        
        # Residual Connection: Match dimensions if needed
        self.use_residual = (in_channels == out_channels and stride == 1)
        self.residual = nn.Identity() if self.use_residual else nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)

    def forward(self, x):
        res = self.residual(x)
        
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.act(x)
        
        return x + res if self.use_residual else x

class RelativePositionalEncoding(nn.Module):
    """Use in AlphaZero-style models"""
    def __init__(self, channels, height=8, width=8):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        # Split channels: c_x + c_y == channels
        c_x = channels // 2         # e.g. for 13 -> 6
        c_y = channels - c_x        # e.g. for 13 -> 7

        # Create learnable embedding tables for relative positions
        self.rel_emb_x = nn.Parameter(torch.randn(width * 2 - 1, c_x))  # Relative X embedding table
        self.rel_emb_y = nn.Parameter(torch.randn(height * 2 - 1, c_y))  # Relative Y embedding table

    def forward(self, x):
        B, C, H, W = x.shape  # Expected: (batch, channels, height, width)
        
        # Create relative coordinate indices
        # Ensure these tensors are on the same device as x
        device = x.device
        coord_x = torch.arange(W, device=device).view(1, -1) - torch.arange(W, device=device).view(-1, 1)  # (W, W)
        coord_y = torch.arange(H, device=device).view(1, -1) - torch.arange(H, device=device).view(-1, 1)  # (H, H)

        # Normalize indices into range [0, 2W-2] and [0, 2H-2]
        coord_x = coord_x + (W - 1)
        coord_y = coord_y + (H - 1)

        # Get relative embeddings from the learned tables and reshape:
        # rel_x: (H, W, c_x)
        rel_x = self.rel_emb_x[coord_x.view(-1)].view(H, W, -1)
        # rel_y: (H, W, c_y)
        rel_y = self.rel_emb_y[coord_y.view(-1)].view(H, W, -1)

        # Concatenate both embeddings along the last dimension so total channels == c_x + c_y == channels
        rel_pos = torch.cat([rel_x, rel_y], dim=-1)  # (H, W, channels)

        # Permute to (1, channels, H, W) for broadcasting over the batch
        rel_pos = rel_pos.permute(2, 0, 1).unsqueeze(0)  # (1, channels, H, W)

        # Add the relative positional encoding to the input features
        return x + rel_pos.to(x.device)


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

    def evaluate(self, one_hots, turns, y):
        """
        Evaluate the model on the data.
        
        :param one_hots: list of one-hot encoded boards
        :param turns: list of turns (0 for White, 1 for Black)
        :param y: list of labels (0 if White wins, 1 if Black wins)
        :return: accuracy
        """
        X, X_t = self.contrastive_learning.transform(one_hots, turns)
        y = torch.tensor(y, dtype=torch.long, device=self.device)

        with torch.no_grad():
            latent = self.feature_extractor(X)
            latent_t = self.feature_extractor(X_t)

            latent = latent.view(latent.shape[0], -1)
            latent_t = latent_t.view(latent_t.shape[0], -1)

            y_pred = self.score_classifier(latent)
            y_pred_t = self.score_classifier(latent_t)

            y_pred = torch.argmax(y_pred, dim=1)
            y_pred_t = torch.argmax(y_pred_t, dim=1)

            acc = (y_pred == y).float().mean().item()
            acc_t = (y_pred_t == y).float().mean().item()

        return (acc + acc_t) / 2
    
    def save(self, element, path):
        if element == "classifier":
            torch.save(self.score_classifier.state_dict(), path)
        elif element == "feature_extractor":
            torch.save(self.feature_extractor.state_dict(), path)
        else:
            raise ValueError("Invalid element, must be 'classifier' or 'feature_extractor'")
        
    def load(self, element, path):
        if element == "classifier":
            self.score_classifier.load_state_dict(torch.load(path, weights_only=True))
        elif element == "feature_extractor":
            self.feature_extractor.load_state_dict(torch.load(path, weights_only=True))
        else:
            raise ValueError("Invalid element, must be 'classifier' or 'feature_extractor'")

    def predict(self, one_hot, turn):
        """
        Predict the outcome of the game given a board state.
        
        :param one_hot: one-hot encoded board
        :param turn: current turn (0 for White, 1 for Black)
        :return: probability distribution over {White win, Black win}
        """
        one_hot = torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)  # Add batch dim
        turn = torch.tensor(turn, dtype=torch.float32).unsqueeze(0)

        X, X_t = self.contrastive_learning.transform(one_hot, turn)

        with torch.no_grad():
            latent = self.feature_extractor(X)
            latent_t = self.feature_extractor(X_t)

            print("1.", latent.shape)
            latent = latent.view(1, -1)  # Flatten
            latent_t = latent_t.view(1, -1)

            # concat latent to have 2 batch (in order to allow normalization)
            latent = torch.cat((latent, latent_t), dim=0)

            print(latent.shape)

            output = self.score_classifier(latent)
            probs = torch.softmax(output, dim=1)

        return probs.squeeze().tolist()

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        return torch.matmul(attention_scores, V)
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x, y):
        Q = self.query(x)
        K = self.key(y)
        V = self.value(y)
        attention_scores = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5))
        return torch.matmul(attention_scores, V)
