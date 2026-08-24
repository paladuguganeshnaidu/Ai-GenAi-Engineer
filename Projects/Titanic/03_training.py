import torch
from torch import nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 1. Load dataset
# ==========================================

data_path = r"Data\titanic\train.csv"

df = pd.read_csv(data_path)


# ==========================================
# 2. Define features and target
# ==========================================

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"

X = df[features].copy()
y = df[target].copy()


# ==========================================
# 3. Handle missing values
# ==========================================

X["Age"] = X["Age"].fillna(
    X["Age"].median()
)

X["Embarked"] = X["Embarked"].fillna(
    X["Embarked"].mode()[0]
)


# ==========================================
# 4. Encode categorical features
# ==========================================

X["Sex"] = X["Sex"].map({
    "male": 0,
    "female": 1
})

X = pd.get_dummies(
    X,
    columns=["Embarked"],
    dtype=int
)


# ==========================================
# 5. Train/Test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 6. Feature scaling
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 7. Convert to PyTorch tensors
# ==========================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train.to_numpy(),
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.float32
)


# ==========================================
# 8. Create TensorDataset
# ==========================================

train_dataset = TensorDataset(
    X_train,
    y_train
)


# ==========================================
# 9. Create DataLoader
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


# ==========================================
# 10. Define MLP
# ==========================================

class TitanicMLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(9, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 11. Create model
# ==========================================

model = TitanicMLP()
# ==========================================
# 12. Loss function
# ==========================================

loss_fn = nn.BCEWithLogitsLoss()


# ==========================================
# 13. Optimizer
# ==========================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01
)


# ==========================================
# 14. Training configuration
# ==========================================

epochs = 10000


# ==========================================
# 15. Training loop
# ==========================================

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        # Clear old gradients
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

        # Add batch loss
        total_loss += loss.item()

    # Average loss for this epoch
    average_loss = total_loss / len(train_loader)

    if epoch%1000==0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {average_loss:.4f}"
        )
# ==========================================
# 16. Save trained model
# ==========================================

model_path = "Projects/Titanic/models/titanic_mlp.pth"

torch.save(
    model.state_dict(),
    model_path
)

print(f"\nModel saved to: {model_path}")