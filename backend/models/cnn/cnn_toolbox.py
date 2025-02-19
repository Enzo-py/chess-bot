import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CBAMChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        max_out = self.fc(self.max_pool(x).view(b, c)).view(b, c, 1, 1)
        out = self.sigmoid(avg_out + max_out)
        return x * out

class CBAMSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # b, c, _, _ = x.size()
        b = x.size(0)
        c = x.size(1)
        scale = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * scale

class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, out_channels, kernel_size, stride=1, padding=0, attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.res_connection = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        self.attention = attention
        if attention:
            self.channel_attention = CBAMChannelAttention(out_channels)
            self.spatial_attention = CBAMSpatialAttention()
    
    def forward(self, x):
        residual = self.res_connection(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x += residual
        x = self.relu(x)
        
        if self.attention:
            x = self.channel_attention(x)
            x = self.spatial_attention(x)
        
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
