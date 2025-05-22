import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
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
        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def _build_sinusoidal_pe(self, length, dim):
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


# === Utility ===
def load_dataset(clean_data_csv: str, error_data_csv: str) -> tuple[pd.DataFrame, pd.DataFrame, torch.Tensor]:
    """Loads the dataset from CSV files."""
    clean_data = pd.read_csv(clean_data_csv, dtype=str).fillna("<pad>")
    error_data = pd.read_csv(error_data_csv, dtype=str).fillna("<pad>")

    label_data = (error_data != clean_data).astype(int).values
    label_data = torch.tensor(label_data, dtype=torch.float32)

    return clean_data, error_data, label_data


def split_dataset(dataframe: pd.DataFrame, label_data: torch.Tensor, tokenizer: Vocab,
                  val_size=0.1, test_size=0.2, random_state=42) -> tuple[tuple[torch.Tensor, torch.Tensor],
tuple[torch.Tensor, torch.Tensor],
tuple[torch.Tensor, torch.Tensor]]:
    """Splits the dataset into train, validation and test sets."""
    token_ids = torch.tensor(tokenizer.encode_table(dataframe), dtype=torch.long)
    num_rows = token_ids.shape[0]

    # First split into train+val and test
    train_val_idx, test_idx = train_test_split(range(num_rows), test_size=test_size, random_state=random_state)

    # Then split train+val into train and val
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_size / (1 - test_size),
                                          random_state=random_state)

    train_tokens = token_ids[train_idx]
    val_tokens = token_ids[val_idx]
    test_tokens = token_ids[test_idx]

    train_labels = label_data[train_idx]
    val_labels = label_data[val_idx]
    test_labels = label_data[test_idx]

    return (train_tokens, train_labels), (val_tokens, val_labels), (test_tokens, test_labels)


def batchify(token_ids: torch.Tensor, batch_size: int = 32):
    num_rows, num_cols = token_ids.shape
    for start_row in range(0, num_rows, batch_size):
        end_row = min(start_row + batch_size, num_rows)
        yield start_row, token_ids[start_row:end_row, :]


# === Training ===
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train(model: nn.Module = None, model_config: dict = None,
          train_data: torch.Tensor = None, val_data: torch.Tensor = None,
          label_data: torch.Tensor = None, val_labels: torch.Tensor = None,
          epochs: int = 50, batch_size: int = 32, lr: float = 1e-3, patience: int = 7,
          device: str = 'cpu') -> nn.Module:
    """
    Train the model and return the best model based on validation loss
    
    Args:
        model: Pre-defined model instance, used if provided
        model_config: Model configuration dictionary, used to create model if model not provided
        train_data: Training data
        val_data: Validation data
        label_data: Training labels
        val_labels: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        patience: Patience value for early stopping
        device: Training device
    
    Returns:
        The best model from training
    """
    # Check required parameters
    if train_data is None or val_data is None or label_data is None or val_labels is None:
        raise ValueError("Training and validation data cannot be None")

    # Create or use model
    if model is None:
        if model_config is None:
            raise ValueError("Either model or model_config must be provided")
        print(f'Creating model using model_config: {model_config}')
        model = TableErrorDetector(**model_config)
    else:
        print(f'Using provided model')

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=patience)

    epoch_losses = []
    val_losses = []
    best_model_state = None
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
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

        avg_train_loss = total_loss / (train_data.shape[0] / batch_size)
        epoch_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for start_row, batch_tokens in batchify(val_data, batch_size):
                batch_size_cur, seq_len = batch_tokens.shape
                row_ids = torch.arange(start_row, start_row + batch_size_cur).unsqueeze(1).expand(batch_size_cur,
                                                                                                  seq_len)
                col_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size_cur, seq_len)

                batch_tokens = batch_tokens.to(device)
                row_ids = row_ids.to(device)
                col_ids = col_ids.to(device)
                target = val_labels[start_row:start_row + batch_size_cur, :].flatten().unsqueeze(0).to(device)

                input_flat = batch_tokens.flatten().unsqueeze(0)
                row_flat = row_ids.flatten().unsqueeze(0)
                col_flat = col_ids.flatten().unsqueeze(0)

                logits = model(input_flat, row_flat, col_flat)
                loss = loss_fn(logits, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / (val_data.shape[0] / batch_size)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Create best model copy
    best_model = None
    if best_model_state is not None:
        if model_config is not None:
            # If model_config is provided, create new model
            best_model = TableErrorDetector(**model_config).to(device)
        else:
            # Otherwise, create a new instance of the same class
            best_model = type(model)().to(device)
        best_model.load_state_dict(best_model_state)
    else:
        best_model = model

    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='o', label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker='o', label='Val Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss,
    }, "models/table_error_detector.pth")

    return best_model


def evaluate(model: nn.Module, token_ids: torch.Tensor, labels: torch.Tensor, batch_size=32, device='cpu',
             lower=0.3, upper=0.7, output_dir="./results"):
    """
    Evaluates the model performance with multiple metrics and visualizations.
    
    Args:
        model: The model to evaluate
        token_ids: Input token IDs
        labels: Ground truth labels
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        lower: Lower threshold for low confidence samples
        upper: Upper threshold for low confidence samples
        output_dir: Directory to save evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    num_rows, num_cols = token_ids.shape
    low_conf_samples = []

    print("\nNow evaluating...\n")
    with torch.no_grad():
        for start_row in range(0, num_rows, batch_size):
            end_row = min(start_row + batch_size, num_rows)
            batch_tokens = token_ids[start_row:end_row].to(device)
            batch_labels = labels[start_row:end_row].to(device)

            batch_size_cur = batch_tokens.size(0)
            seq_len = batch_tokens.size(1)

            row_ids = torch.arange(start_row, end_row, device=device).unsqueeze(1).expand(batch_size_cur, seq_len)
            col_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size_cur, seq_len)

            input_flat = batch_tokens.flatten().unsqueeze(0)
            row_flat = row_ids.flatten().unsqueeze(0)
            col_flat = col_ids.flatten().unsqueeze(0)

            logits = model(input_flat, row_flat, col_flat)
            probs = torch.sigmoid(logits).squeeze(0)

            # Collect low confidence predictions
            mask = (probs > lower) & (probs < upper)
            indices = mask.nonzero(as_tuple=False)
            for idx in indices:
                global_idx = start_row * seq_len + idx.item()
                low_conf_samples.append((global_idx, probs[idx].item()))

            preds = (probs > 0.5).int().cpu().numpy()
            probs = probs.cpu().numpy()
            labels_np = batch_labels.cpu().numpy().flatten()

            all_preds.append(preds)
            all_probs.append(probs)
            all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    # Calculate and print low confidence error rate
    error_rate = check_low_conf_error_rate(low_conf_samples, labels)
    print(f"\nLow Confidence Error Rate: {error_rate:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Plot Precision-Recall curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)

    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_curves.png'))
    plt.close()

    # Save evaluation metrics
    metrics = {
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'low_conf_error_rate': error_rate
    }

    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")

    return all_preds, all_probs, all_labels, metrics


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
    # dataset_name = 'flights'
    # dataset_name = 'movies'
    # dataset_name = 'billionaire'
    # dataset_name = 'beers'
    dataset_name = 'hospital'
    # dataset_name = 'rayyan'

    error_df_csv = f'./data/{dataset_name}_error-01.csv'
    clean_df_csv = f'./data/{dataset_name}_clean.csv'

    clean_df, error_df, labels = load_dataset(clean_df_csv, error_df_csv)

    vocab = Vocab()
    train_set, val_set, test_set = split_dataset(error_df, labels, vocab, val_size=0.2, test_size=0.4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Model configuration
    model_config = {
        'vocab_size': len(vocab),
        'max_rows': error_df.shape[0],
        'max_cols': error_df.shape[1],
        'embed_dim': 256,
        'num_heads': 8,
        'num_layers': 4,
        'dropout': 0.2,
        'hidden_dim': 512,
        'position_encoding': 'sinusoidal'
    }

    # Create and train model
    model = train(
        model_config=model_config,
        train_data=train_set[0],
        val_data=val_set[0],
        label_data=train_set[1],
        val_labels=val_set[1],
        epochs=100,
        batch_size=128 * 4,
        lr=5e-4,
        patience=20,
        device=device
    )

    # Evaluate model
    evaluate(model, test_set[0], test_set[1], batch_size=128 * 4, device=device)
