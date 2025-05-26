import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

from model import Vocab, TableErrorDetector


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
    print(train_labels.sum())
    print(val_labels.sum())
    print(test_labels.sum())

    return (train_tokens, train_labels), (val_tokens, val_labels), (test_tokens, test_labels)


def batchify(token_ids: torch.Tensor, batch_size: int = 32):
    num_rows, num_cols = token_ids.shape
    for start_row in range(0, num_rows, batch_size):
        end_row = min(start_row + batch_size, num_rows)
        yield start_row, token_ids[start_row:end_row, :]


def batchify_shuffled_with_labels(token_ids, token_labels: torch.Tensor, batch_size):
    num_rows = token_ids.size(0)
    indices = torch.randperm(num_rows)
    print('indices:', len(indices), indices[:10])  # Print first 10 indices for debugging

    shuffled_tokens = token_ids[indices]
    shuffled_labels = token_labels[indices]

    for i in range(0, num_rows, batch_size):
        yield (
            shuffled_tokens[i:i + batch_size],
            shuffled_labels[i:i + batch_size],
            indices[i:i + batch_size],  # keep for row_ids
        )


# === Training ===
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        if patience < 0:
            raise ValueError("Patience must be non-negative")
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
            if self.counter >= self.patience > 0:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train(model: nn.Module | None, model_config: dict, train_data: torch.Tensor, val_data: torch.Tensor,
          train_labels: torch.Tensor, val_labels: torch.Tensor, epochs: int = 50, batch_size: int = 32,
          lr: float = 1e-3, patience: int = 0, device: str = 'cpu') -> nn.Module:
    """
    Train the model and return the best model based on validation loss
    
    Args:
        model: Pre-defined model instance, used if provided
        model_config: Model configuration dictionary, used to create model if model not provided
        train_data: Training data
        val_data: Validation data
        train_labels: Training labels
        val_labels: Validation labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        patience: Patience value for early stopping, 0 means no early stopping
        device: Training device
    
    Returns:
        The best model from training
    """
    # Check required parameters
    if train_data is None or val_data is None or train_labels is None or val_labels is None:
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

    # optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = nn.BCEWithLogitsLoss()

    # 更推荐使用 AdamW，支持权重衰减，可提高泛化能力
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Warmup + Cosine Annealing 更适合大模型训练
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # 每隔多少 epoch 重启一次
        T_mult=2,  # 重启时周期扩大倍数
        eta_min=1e-6  # 最小学习率
    )

    early_stopping = EarlyStopping(patience=patience)

    epoch_losses = []
    val_losses = []
    best_model_state = None
    best_val_loss = float('inf')

    num_rows, num_cols = train_data.shape
    # total_samples = 3 * num_rows
    # num_batches = (total_samples + batch_size - 1) // batch_size  # 向上取整
    # batch_sampler = batchify_random(train_data, batch_size)
    # print('Total samples:', total_samples, 'Number of batches:', num_batches, 'batch size:', batch_size, 'num_rows:', num_rows)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # for start_row, batch_tokens in batchify(train_data, batch_size):
        for batch_tokens, batch_labels, batch_rows in batchify_shuffled_with_labels(train_data, train_labels,
                                                                                    batch_size):
            batch_size_cur, seq_len = batch_tokens.shape  # current batch size and number of columns
            row_ids = batch_rows.unsqueeze(1).expand(-1, seq_len)  # (B, C)
            col_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size_cur, seq_len)  # (B, C)

            # Move to device
            batch_tokens = batch_tokens.to(device)
            batch_labels = batch_labels.to(device)
            row_ids = row_ids.to(device)
            col_ids = col_ids.to(device)

            # Flatten for model input
            input_flat = batch_tokens.flatten().unsqueeze(0)  # (1, B*C)
            row_flat = row_ids.flatten().unsqueeze(0)
            col_flat = col_ids.flatten().unsqueeze(0)
            target = batch_labels.flatten().unsqueeze(0)  # (1, B*C)

            # Forward and loss
            logits = model(input_flat, row_flat, col_flat)
            loss = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / (train_data.shape[0] / batch_size)
        epoch_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_tokens, batch_labels, batch_rows in batchify_shuffled_with_labels(val_data, val_labels,
                                                                                        batch_size):
                batch_size_cur, seq_len = batch_tokens.shape

                row_ids = batch_rows.unsqueeze(1).expand(-1, seq_len)  # (B, C)
                col_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size_cur, seq_len)  # (B, C)

                batch_tokens = batch_tokens.to(device)
                batch_labels = batch_labels.to(device)
                row_ids = row_ids.to(device)
                col_ids = col_ids.to(device)

                input_flat = batch_tokens.flatten().unsqueeze(0)  # (1, B*C)
                row_flat = row_ids.flatten().unsqueeze(0)
                col_flat = col_ids.flatten().unsqueeze(0)
                target = batch_labels.flatten().unsqueeze(0)  # (1, B*C)

                logits = model(input_flat, row_flat, col_flat)
                loss = loss_fn(logits, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / (val_data.shape[0] / batch_size)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            print('New best model found at epoch', epoch + 1, 'with validation loss:', avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Learning rate scheduling
        # scheduler.step(avg_val_loss)

        # Early stopping
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Create best model copy
    # best_model = None
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
    dataset_name = 'movies'
    # dataset_name = 'billionaire'
    # dataset_name = 'beers'
    # dataset_name = 'hospital'
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
        model=None,  # Use None to create a new model
        model_config=model_config,
        train_data=train_set[0],
        val_data=val_set[0],
        train_labels=train_set[1],
        val_labels=val_set[1],
        epochs=100,
        batch_size=256 * 2,
        lr=1e-3,
        patience=30,
        device=device
    )

    # Evaluate model
    evaluate(model, test_set[0], test_set[1], batch_size=256 * 2, device=device)
