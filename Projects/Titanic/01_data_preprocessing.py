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


# ==========================================
# 8. Check final data
# ==========================================

print("\nX_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("\ny_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

print("\nFirst training sample:")
print(X_train[0])

print("\nFirst target:")
print(y_train[0])