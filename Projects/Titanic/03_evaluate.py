import torch
from torch import nn
from pathlib import Path

from src.data import get_data
from src.model import TitanicMLP
from src.utils import load_model

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================
# 1. Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "titanic_mlp.pth"
)


# ==========================================
# 2. Load processed data
# ==========================================

X_train, X_val, X_test, y_train, y_val, scaler = get_data()

print("Validation data:", X_val.shape)
print("Validation targets:", y_val.shape)


# ==========================================
# 3. Create model
# ==========================================

model = TitanicMLP()

print("\nModel:")
print(model)


# ==========================================
# 4. Load trained model
# ==========================================

model = load_model(
    model,
    MODEL_PATH
)


# ==========================================
# 5. Evaluation mode
# ==========================================

model.eval()


# ==========================================
# 6. Loss function
# ==========================================

loss_fn = nn.BCEWithLogitsLoss()


# ==========================================
# 7. Make predictions
# ==========================================

with torch.no_grad():

    # Forward pass
    logits = model(X_val)

    # Validation loss
    loss = loss_fn(
        logits.squeeze(),
        y_val
    )

    # Convert logits to probabilities
    probabilities = torch.sigmoid(logits)

    # Convert probabilities to predictions
    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.squeeze()


# ==========================================
# 8. Convert tensors to NumPy
# ==========================================

y_true = y_val.numpy()

y_pred = predictions.numpy()


# ==========================================
# 9. Calculate accuracy
# ==========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


# ==========================================
# 10. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred
)


# ==========================================
# 11. Precision
# ==========================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


# ==========================================
# 12. Recall
# ==========================================

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


# ==========================================
# 13. F1 score
# ==========================================

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ==========================================
# 14. Display results
# ==========================================

print("\nEvaluation Results:")
print("-------------------")

print(f"Validation Loss: {loss.item():.4f}")
print(f"Accuracy:         {accuracy:.4f}")
print(f"Accuracy:         {accuracy * 100:.2f}%")

print(f"Precision:        {precision:.4f}")
print(f"Recall:           {recall:.4f}")
print(f"F1 Score:         {f1:.4f}")


# ==========================================
# 15. Display confusion matrix
# ==========================================

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 16. Display prediction examples
# ==========================================

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_true[:10])

print("\nFirst 10 Probabilities:")
print(
    probabilities[:10].squeeze()
)