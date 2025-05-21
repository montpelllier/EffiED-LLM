import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# === Vocab ===
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

    def encode_table(self, dataframe: pd.DataFrame):
        """Encodes a table into a list of integer IDs."""
        encoded = []
        for row in dataframe.itertuples(index=False):
            encoded.append([self.encode(str(cell)) for cell in row])
        return encoded


# === Model ===
class TableErrorDetector(nn.Module):
    def __init__(self, vocab_size, max_rows, max_cols, embed_dim=128, num_heads=4, num_layers=4):
        """Initializes the TableErrorDetector transformer model."""
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim,
            num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        # Register fixed PE buffers
        self.register_buffer("row_pe", self._build_sinusoidal_pe(max_rows, embed_dim))
        self.register_buffer("col_pe", self._build_sinusoidal_pe(max_cols, embed_dim))

    def _build_sinusoidal_pe(self, length, dim):
        """Generates sinusoidal PE matrix of shape [length, dim]"""
        pe = torch.zeros(length, dim)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # fill sin(x) in even number position
        pe[:, 1::2] = torch.cos(position * div_term)  # fill cos(x) in odd number position
        return pe  # shape: [length, dim]

    def forward(self, token_ids, row_ids, col_ids):
        x = self.token_embed(token_ids)
        # Add 2D sinusoidal PE
        row_pe = self.row_pe[row_ids]  # [B, N, D]
        col_pe = self.col_pe[col_ids]  # [B, N, D]
        x = x + row_pe + col_pe
        x = self.norm(x)
        x = self.encoder(x)
        logits = self.classifier(x).squeeze(-1)
        return logits


# === Utility ===
def load_dataset(clean_data_csv: str, error_data_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """Loads the dataset from CSV files."""
    clean_data = pd.read_csv(clean_data_csv, dtype=str).fillna("<pad>")
    error_data = pd.read_csv(error_data_csv, dtype=str).fillna("<pad>")

    label_data = (error_data != clean_data).astype(int).values
    label_data = torch.tensor(label_data, dtype=torch.float32)

    return clean_data, error_data, label_data


def split_dataset(dataframe: pd.DataFrame, label_datat: torch.Tensor, tokenizer: Vocab, test_size=0.2,
                  random_state=42) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Splits the dataset into train and test sets."""
    token_ids = torch.tensor(tokenizer.encode_table(dataframe), dtype=torch.long)  # [rows, cols]
    num_rows = token_ids.shape[0]

    train_idx, test_idx = train_test_split(range(num_rows), test_size=test_size, random_state=random_state)

    train_tokens = token_ids[train_idx]
    test_tokens = token_ids[test_idx]
    train_labels = label_datat[train_idx]
    test_labels = label_datat[test_idx]

    return (train_tokens, train_labels), (test_tokens, test_labels)


def batchify(token_ids: torch.Tensor, batch_size: int = 32):
    num_rows, num_cols = token_ids.shape
    for start_row in range(0, num_rows, batch_size):
        end_row = min(start_row + batch_size, num_rows)
        yield start_row, token_ids[start_row:end_row, :]


# === Training ===
def train(train_data: torch.Tensor, label_data: torch.Tensor, vocab_size: int, shape: tuple[int, int], epochs=50,
          batch_size=32, lr=1e-3, device='cpu') -> TableErrorDetector:
    print(f'vocab_size: {vocab_size}')
    print(f'num_rows: {shape[0]}')
    print(f'num_cols: {shape[1]}')

    model = TableErrorDetector(vocab_size=vocab_size, max_rows=shape[0], max_cols=shape[1],
                               embed_dim=128, num_heads=4, num_layers=4)
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for start_row, batch_tokens in batchify(train_data, batch_size):
            batch_size_cur, seq_len = batch_tokens.shape
            row_ids = torch.arange(start_row, start_row + batch_size_cur).unsqueeze(1).expand(batch_size_cur, seq_len)
            col_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size_cur, seq_len)

            batch_tokens = batch_tokens.to(device)
            row_ids = row_ids.to(device)
            col_ids = col_ids.to(device)
            target = label_data[start_row:start_row + batch_size_cur, :].flatten().unsqueeze(0).to(device)

            input_flat = batch_tokens.flatten().unsqueeze(0)
            row_flat = row_ids.flatten().unsqueeze(0)
            col_flat = col_ids.flatten().unsqueeze(0)

            logits = model(input_flat, row_flat, col_flat)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (shape[0] / batch_size)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} - Total Loss: {total_loss:.4f}, Avg Loss: {avg_loss:.4f}")

    # draw loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/table_error_detector.pth")

    return model


def evaluate(model: nn.Module, token_ids: torch.Tensor, labels: torch.Tensor, batch_size=32, device='cpu', lower=0.3,
             upper=0.7):
    model.eval()
    all_preds = []
    all_labels = []

    num_rows, num_cols = token_ids.shape
    low_conf_samples = []

    print("\nNow evaluating...\n")
    with torch.no_grad():
        for start_row in range(0, num_rows, batch_size):
            end_row = min(start_row + batch_size, num_rows)
            batch_tokens = token_ids[start_row:end_row].to(device)  # [batch, cols]
            batch_labels = labels[start_row:end_row].to(device)

            batch_size_cur = batch_tokens.size(0)
            seq_len = batch_tokens.size(1)

            row_ids = torch.arange(start_row, end_row, device=device).unsqueeze(1).expand(batch_size_cur, seq_len)
            col_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size_cur, seq_len)

            input_flat = batch_tokens.flatten().unsqueeze(0)  # [1, batch*seq]
            row_flat = row_ids.flatten().unsqueeze(0)
            col_flat = col_ids.flatten().unsqueeze(0)

            logits = model(input_flat, row_flat, col_flat)  # [1, batch*seq, 1]
            probs = torch.sigmoid(logits).squeeze(0)

            mask = (probs > lower) & (probs < upper)
            indices = mask.nonzero(as_tuple=False)
            for idx in indices:
                global_idx = start_row * seq_len + idx.item()  # 全局索引，或者根据需要转换成二维索引
                low_conf_samples.append((global_idx, probs[idx].item()))

            preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy().flatten()
            labels_np = batch_labels.cpu().numpy().flatten()

            all_preds.append(preds)
            all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    print(classification_report(all_labels, all_preds, digits=4))
    check_low_conf_error_rate(low_conf_samples, labels)

    return all_preds, all_labels


def check_low_conf_error_rate(low_conf_samples, labels):
    total = len(low_conf_samples)
    if total == 0:
        print("no samples found")
        return 0

    error_count = 0
    for global_idx, prob in low_conf_samples:
        # global_idx is flatern indexes, transform back to 2-dim indexes
        num_cols = labels.shape[1]
        row = global_idx // num_cols
        col = global_idx % num_cols

        true_label = labels[row, col].item()
        pred_label = 1 if prob > 0.5 else 0

        if pred_label != true_label:
            error_count += 1

    error_rate = error_count / total
    print(f"Total low confidence sample: {total}, error num: {error_count}, error rate: {error_rate:.4f}")
    return error_rate


if __name__ == '__main__':
    dataset_name = 'flights'
    # dataset_name = 'movies'
    # dataset_name = 'billionaire'
    # dataset_name = 'beers'
    # dataset_name = 'hospital'
    # dataset_name = 'rayyan'

    error_df_csv = f'./data/{dataset_name}_error-01.csv'
    clean_df_csv = f'./data/{dataset_name}_clean.csv'

    clean_df, error_df, labels = load_dataset(clean_df_csv, error_df_csv)

    vocab = Vocab()
    train_set, test_set = split_dataset(error_df, labels, vocab, 0.3)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = train(train_set[0], train_set[1], vocab_size=len(vocab), shape=error_df.shape, epochs=100,
                  batch_size=128 * 4, lr=1e-4, device=device)
    evaluate(model, test_set[0], test_set[1], batch_size=128 * 4, device=device)
