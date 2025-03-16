import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEmbedding(nn.Module):
    """
    Chess-specific positional embedding that encodes the board position information.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        # Create embeddings for rank (0-7) and file (0-7)
        self.rank_embedding = nn.Embedding(8, embedding_dim // 2)
        self.file_embedding = nn.Embedding(8, embedding_dim // 2)
        
    def forward(self, x):
        # x is expected to be of shape [batch, seq_len, features]
        # where seq_len = 64 (8x8 board flattened)
        batch_size = x.size(0)
        
        # Create position indices for an 8x8 board
        ranks = torch.repeat_interleave(torch.arange(8), 8).unsqueeze(0).to(x.device)
        files = torch.tile(torch.arange(8), (8,)).unsqueeze(0).to(x.device)
        
        # Get embeddings
        rank_emb = self.rank_embedding(ranks)  # [1, 64, embed_dim//2]
        file_emb = self.file_embedding(files)  # [1, 64, embed_dim//2]
        
        # Combine for final positional embedding
        pos_emb = torch.cat([rank_emb, file_emb], dim=-1)  # [1, 64, embed_dim]
        pos_emb = pos_emb.expand(batch_size, -1, -1)  # [batch, 64, embed_dim]
        
        return x + pos_emb


class SelfAttention(nn.Module):
    """
    Self-attention module with multi-head attention.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # Apply layer normalization before attention (pre-norm transformer)
        normalized_x = self.norm(x)
        # Self-attention
        attn_output, _ = self.multihead_attn(normalized_x, normalized_x, normalized_x)
        # Residual connection
        return x + attn_output


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Apply layer normalization before the feed-forward network
        normalized_x = self.norm(x)
        # Two-layer feed-forward network with ReLU activation
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(normalized_x))))
        # Residual connection
        return x + ff_output


class TransformerBlock(nn.Module):
    """
    Transformer block with self-attention and feed-forward network.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout)
        
    def forward(self, x):
        x = self.attention(x)
        x = self.feed_forward(x)
        return x


class PieceEmbedding(nn.Module):
    """
    Embedding layer for chess pieces.
    """
    def __init__(self, embedding_dim):
        super().__init__()
        # 13 piece types: 6 white, 6 black, 1 empty
        self.piece_embedding = nn.Embedding(13, embedding_dim)
        
    def forward(self, one_hot_pieces):
        # one_hot_pieces: [batch, 8, 8, 13]
        batch_size = one_hot_pieces.size(0)
        
        # Convert one-hot to indices
        piece_indices = torch.argmax(one_hot_pieces, dim=-1)  # [batch, 8, 8]
        
        # Flatten the board for embedding lookup
        flat_indices = piece_indices.reshape(batch_size, -1)  # [batch, 64]
        
        # Get embeddings
        embeddings = self.piece_embedding(flat_indices)  # [batch, 64, embedding_dim]
        
        return embeddings


class ChessCrossAttention(nn.Module):
    """
    Cross-attention module for chess-specific features.
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, key_value):
        # Apply layer normalization
        normalized_query = self.norm1(query)
        normalized_kv = self.norm2(key_value)
        
        # Cross-attention
        attn_output, _ = self.multihead_attn(normalized_query, normalized_kv, normalized_kv)
        
        # Residual connection
        return query + attn_output 