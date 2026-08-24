import torch
from torch import nn
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

X["Age"] = X["Age"].fillna(X["Age"].median())

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
# 7. Convert to float32
# ==========================================

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

y_train = y_train.to_numpy(dtype="float32")
y_test = y_test.to_numpy(dtype="float32")


X_train = X_train.astype("float32")
X_test = X_test.astype("float32")

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)



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
model = TitanicMLP()
# ==========================================
# Forward pass with one real passenger
# ==========================================

sample = X_train[0]

print("\nSample:")
print(sample)

print("\nSample shape:")
print(sample.shape)

output = model(sample)

print("\nModel output:")
print(output)

print("\nOutput shape:")
print(output.shape)
# ==========================================
# Inspect each layer
# ==========================================

x = sample

linear_output = model.network[0](x)

print("\nAfter Linear(9, 8):")
print(linear_output)
print("Shape:", linear_output.shape)

relu_output = model.network[1](linear_output)

print("\nAfter ReLU:")
print(relu_output)
print("Shape:", relu_output.shape)

final_output = model.network[2](relu_output)

print("\nAfter Linear(8, 1):")
print(final_output)
print("Shape:", final_output.shape)