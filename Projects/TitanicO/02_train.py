import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from src.data import get_data
from src.model import TitanicMLP
from src.utils import save_model

# ==========================================
# Reproducibility
# ==========================================

torch.manual_seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent


# ==========================================
# 1. Load processed data
# ==========================================

X_train, X_val, X_test, y_train, y_val, scaler = get_data()

print("Training data:", X_train.shape)
print("Validation data:", X_val.shape)
print("Test data:", X_test.shape)
print("Training targets:", y_train.shape)
print("Validation targets:", y_val.shape)

# ==========================================
# 2. Create training dataset
# ==========================================

train_dataset = TensorDataset(
    X_train,
    y_train
)


# ==========================================
# 3. Create DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# ==========================================
# 4. Create model
# ==========================================

model = TitanicMLP()

print("\nModel:")
print(model)


# ==========================================
# 5. Loss function
# ==========================================

loss_fn = nn.BCEWithLogitsLoss()


# ==========================================
# 6. Optimizer
# ==========================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 7. Training configuration
# ==========================================

epochs = 7_800


# ==========================================
# 8. Training loop
# ==========================================

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for X_batch, y_batch in train_loader:

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(X_batch)

        # Calculate loss
        loss = loss_fn(
            output.squeeze(),
            y_batch
        )

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        # Track loss
        total_loss += loss.item()

    # Average loss for this epoch

    average_loss = (
        total_loss / len(train_loader)
    )

    # Print every 1000 epochs

    if epoch%100 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )


# ==========================================
# 9. Save trained model
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent
model_path = PROJECT_ROOT / "models" / "titanic_mlp.pth"

save_model(
    model,
    model_path
)