import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, save_every=10, save_dir="saved_models", save_name="scTRaCT_checkpoint"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        y_true = []
        y_pred = []

        loop = tqdm(train_loader, leave=False, desc=f"Training Epoch {epoch}")
        for X_counts_batch, X_dist_batch, y_batch in loop:
            X_counts_batch, X_dist_batch, y_batch = X_counts_batch.to(device), X_dist_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_counts_batch, X_dist_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average="macro")

        # Print nicely
        print(f"[ Train | {epoch:03d}/{num_epochs:03d} ] loss = {avg_loss:.5f}, acc = {train_acc:.5f}, f1 = {train_f1:.5f}")

        # Save checkpoint
        if epoch % save_every == 0:
            save_path = os.path.join(save_dir, f"{save_name}_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at {save_path}")


def evaluate_model(model, test_loader, label_encoder):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_counts_batch, X_dist_batch, y_batch in test_loader:
            X_counts_batch, X_dist_batch = X_counts_batch.to(device), X_dist_batch.to(device)
            outputs = model(X_counts_batch, X_dist_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(y_batch.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = accuracy_score(all_labels.numpy(), all_preds.numpy())
    f1 = f1_score(all_labels.numpy(), all_preds.numpy(), average="macro")
    
    # Convert predictions back to cell type names
    predicted_celltypes = label_encoder.inverse_transform(all_preds.numpy())

    return acc, f1, predicted_celltypes




def evaluate_on_query(
    adata,
    checkpoint_path,
    label_encoder,
    lognorm_layer="lognorm",
    distance_layer="distance_matrix",
    cell_type_key="cell_type",
    is_train_key="is_train",
    num_heads=8,
    dim_feedforward=2048,
    dropout=0.1,
    embedding_dim=1024,
    batch_size=64
):
    """
    Evaluate a trained scTRaCT model on the query (test) set.

    Args:
        adata: AnnData object containing both train and test samples.
        checkpoint_path: Path to the saved model checkpoint (.pth).
        label_encoder: LabelEncoder fitted on training labels.
        lognorm_layer: Name of the log-normalized layer in adata.
        distance_layer: Name of the distance matrix layer in adata.
        cell_type_key: Column in .obs containing true cell type labels.
        is_train_key: Column in .obs to distinguish train/test samples.
        num_heads: Number of attention heads.
        dim_feedforward: Size of feedforward network.
        dropout: Dropout rate.
        embedding_dim: Embedding dimension.
        batch_size: Batch size for evaluation.

    Returns:
        Accuracy and macro F1 score on the query set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract query/test data
    query_adata = adata[adata.obs[is_train_key] != 'train'].copy()

    X_counts_query = query_adata.layers[lognorm_layer].toarray()
    X_dist_query = query_adata.layers[distance_layer]
    if hasattr(X_dist_query, "toarray"):
        X_dist_query = X_dist_query.toarray()

    epsilon = 1e-6
    X_dist_query = 1 / (X_dist_query + epsilon)

    X_counts_query_tensor = torch.tensor(X_counts_query, dtype=torch.float32)
    X_dist_query_tensor = torch.tensor(X_dist_query, dtype=torch.float32)

    # Ground truth labels
    y_true = query_adata.obs[cell_type_key].values
    y_true_encoded = label_encoder.transform(y_true)
    y_true_tensor = torch.tensor(y_true_encoded, dtype=torch.long)

    # Create DataLoader
    query_dataset = TensorDataset(X_counts_query_tensor, X_dist_query_tensor, y_true_tensor)
    query_loader = DataLoader(query_dataset, batch_size=batch_size)

    # Initialize model
    num_genes = X_counts_query.shape[1]
    num_classes = len(label_encoder.classes_)

    from scTRaCT.model import TransformerModel  
    from scTRaCT.trainer import evaluate_model

    model = TransformerModel(
        num_genes=num_genes,
        num_classes=num_classes,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        embedding_dim=embedding_dim
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    # Evaluate
    acc, f1, predicted_celltypes = evaluate_model(model, query_loader, label_encoder)


    print(f"Test/Query Set Accuracy: {acc:.4f}")
    print(f"Test/Query Set F1 Score: {f1:.4f}")

    return acc, f1, predicted_celltypes, query_adata