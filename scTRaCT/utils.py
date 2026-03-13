import torch
from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(y_true, y_pred):
    """
    Computes accuracy and macro-F1 score.

    Args:
        y_true: True labels (numpy or torch tensor)
        y_pred: Predicted labels (numpy or torch tensor)

    Returns:
        Dictionary with accuracy and F1 score
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    return {"accuracy": acc, "f1_score": f1}
