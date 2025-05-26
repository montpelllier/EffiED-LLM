# === Vocab ===
import math

import pandas
import torch
from torch import nn


class Vocab(dict):
    def __init__(self):
        """Initializes the Vocab object."""
        super().__init__()
        self['<pad>'] = 0  # padding token id
        self['<unk>'] = 1  # unknown token id
        self.id2token = {v: k for k, v in self.items()}

    def __missing__(self, key):
        return self['<unk>']

    def encode(self, token: str):
        """Encodes a token into an integer ID."""
        if token not in self:
            idx = len(self)
            self[token] = idx
            self.id2token[idx] = token
        return self[token]

    def encode_table(self, dataframe: pandas.DataFrame):
        """Encodes a table into a list of integer IDs."""
        encoded = []
        for row in dataframe.itertuples(index=False):
            encoded.append([self.encode(str(cell)) for cell in row])
        return encoded


# === Model ===
class RowColMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.col_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, row_mask=None, col_mask=None):
        # x shape: (B, R, C, D)
        B, R, C, D = x.shape

        # 行内注意力（按行做注意力，序列长度是C）
        row_seq = x.reshape(B * R, C, D)  # (B*R, C, D)
        row_out, _ = self.row_attn(row_seq, row_seq, row_seq, key_padding_mask=row_mask)  # (B*R, C, D)
        row_out = row_out.reshape(B, R, C, D)

        # 列内注意力（按列做注意力，序列长度是R）
        col_seq = x.permute(0, 2, 1, 3).reshape(B * C, R, D)  # (B*C, R, D)
        col_out, _ = self.col_attn(col_seq, col_seq, col_seq, key_padding_mask=col_mask)  # (B*C, R, D)
        col_out = col_out.reshape(B, C, R, D).permute(0, 2, 1, 3)  # (B, R, C, D)

        # 融合并残差连接
        combined = torch.cat([row_out, col_out], dim=-1)  # (B, R, C, 2*D)
        combined = self.linear(combined)
        combined = self.dropout(combined)
        out = self.norm(x + combined)  # 残差连接

        return out


class TableErrorDetector(nn.Module):
    def __init__(self, vocab_size, max_rows, max_cols, embed_dim=128, num_heads=4, num_layers=4,
                 dropout=0.1, hidden_dim=None, position_encoding='sinusoidal'):
        """Initializes the TableErrorDetector transformer model.

        Args:
            vocab_size: Size of the vocabulary
            max_rows: Maximum number of rows in input tables
            max_cols: Maximum number of columns in input tables
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            hidden_dim: Hidden dimension in feed forward network (default: 4*embed_dim)
            position_encoding: Type of position encoding ('sinusoidal' or 'learned')
        """
        super().__init__()
        self.embed_dim = embed_dim
        hidden_dim = hidden_dim or embed_dim * 4

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Position encoding
        self.position_encoding = position_encoding
        if position_encoding == 'sinusoidal':
            self.register_buffer("row_pe", self._build_sinusoidal_pe(max_rows, embed_dim))
            self.register_buffer("col_pe", self._build_sinusoidal_pe(max_cols, embed_dim))
        else:  # learned
            self.row_pe = nn.Parameter(torch.randn(max_rows, embed_dim))
            self.col_pe = nn.Parameter(torch.randn(max_cols, embed_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.layers = nn.ModuleList([
        #     RowColMultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        #     for _ in range(num_layers)
        # ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    @classmethod
    def _build_sinusoidal_pe(cls, length, dim):
        """Generates sinusoidal PE matrix of shape [length, dim]"""
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, token_ids, row_ids, col_ids):
        x = self.token_embed(token_ids)

        # Add position encodings
        row_pe = self.row_pe[row_ids]
        col_pe = self.col_pe[col_ids]
        x = x + row_pe + col_pe

        # Apply dropout after embeddings and position encoding
        x = self.dropout(x)

        # Layer norm and transformer
        x = self.norm(x)
        x = self.encoder(x)

        # Classification
        logits = self.classifier(x).squeeze(-1)
        return logits

    # def forward(self, token_ids, row_ids, col_ids):
    #     # token_ids: (B, C) —— 每个样本是一个“整行”有C列
    #     B, C = token_ids.shape
    #     device = token_ids.device
    #
    #     # 词嵌入
    #     x = self.token_embed(token_ids)  # (B, C, D)
    #
    #     # 行列位置编码
    #     row_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, C)  # (B, C)
    #     col_ids = torch.arange(C, device=device).unsqueeze(0).expand(B, C)  # (B, C)
    #     row_pe = self.row_pe[row_ids]  # (B, C, D)
    #     col_pe = self.col_pe[col_ids]  # (B, C, D)
    #
    #     x = x + row_pe + col_pe  # (B, C, D)
    #     x = self.dropout(x)
    #
    #     # reshape 为 (B, R, C, D)，这里 R 实际就是 B，换一个名字更清晰
    #     x = x.unsqueeze(1)  # → (B, 1, C, D)，即每个 batch 只有一行
    #
    #     # 多层行列解耦注意力
    #     for layer in self.layers:
    #         x = layer(x)  # 每层处理 (B, R=1, C, D)
    #
    #     x = self.norm(x)  # (B, 1, C, D)
    #     x = x.squeeze(1)  # → (B, C, D)
    #
    #     # 送入分类器，每个 cell 一个预测值
    #     logits = self.classifier(x).squeeze(-1)  # (B, C)
    #     return logits