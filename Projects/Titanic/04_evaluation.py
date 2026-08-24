import torch
from torch import nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# 1. Define the SAME MLP architecture
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
# 2. Create model
# ==========================================

model = TitanicMLP()


# ==========================================
# 3. Load trained model parameters
# ==========================================

model_path = "Projects/Titanic/models/titanic_mlp.pth"

model.load_state_dict(
    torch.load(
        model_path,
        weights_only=True
    )
)

print("Trained model loaded successfully.")


# ==========================================
# 4. Switch model to evaluation mode
# ==========================================

model.eval()

print("Model is in evaluation mode.")


# ==========================================
# 5. Load dataset
# ==========================================

data_path = r"Data\titanic\train.csv"

df = pd.read_csv(data_path)


# ==========================================
# 6. Define features and target
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
# 7. Handle missing values
# ==========================================

X["Age"] = X["Age"].fillna(
    X["Age"].median()
)

X["Embarked"] = X["Embarked"].fillna(
    X["Embarked"].mode()[0]
)


# ==========================================
# 8. Encode categorical features
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
# 9. Recreate SAME train/test split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 10. Apply feature scaling
# ==========================================

scaler = StandardScaler()

# Fit ONLY on training data
X_train = scaler.fit_transform(X_train)

# Use the same fitted scaler on test data
X_test = scaler.transform(X_test)


# ==========================================
# 11. Convert test data to PyTorch tensors
# ==========================================

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test.to_numpy(),
    dtype=torch.float32
)

print("\nTest data shape:")
print(X_test.shape)

print("\nTest target shape:")
print(y_test.shape)


# ==========================================
# 12. Inference using trained model
# ==========================================

with torch.no_grad():

    logits = model(X_test)


print("\nLogits shape:")
print(logits.shape)

print("\nFirst 10 logits:")
print(logits[:10])


# ==========================================
# 13. Convert logits to probabilities
# ==========================================

probabilities = torch.sigmoid(logits)

print("\nFirst 10 probabilities:")
print(probabilities[:10])


# ==========================================
# 14. Convert probabilities to predictions
# ==========================================

predictions = (
    probabilities >= 0.5
).float()

predictions = predictions.squeeze()


print("\nFirst 10 predictions:")
print(predictions[:10])


# ==========================================
# 15. Calculate accuracy
# ==========================================

correct = (
    predictions == y_test
).sum().item()

total = len(y_test)

accuracy = correct / total

print("\nEvaluation Results:")
print("-------------------")

print("Correct predictions:", correct)
print("Total test samples:", total)

print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")