import pandas as pd
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==========================================
# Project paths
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "test.csv"


# ==========================================
# Feature configuration
# ==========================================

FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

TARGET = "Survived"


# ==========================================
# Load raw datasets
# ==========================================

def load_data():
    """
    Load Titanic training and test datasets.
    """

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


# ==========================================
# Prepare raw features
# ==========================================

def preprocess_features(train_df, test_df):
    """
    Clean and encode train/test features.

    Returns:
        X_train
        X_test
    """

    # --------------------------------------
    # Select features
    # --------------------------------------

    X_train = train_df[FEATURES].copy()
    X_test = test_df[FEATURES].copy()

    # --------------------------------------
    # Handle missing Age
    # --------------------------------------

    age_median = X_train["Age"].median()

    X_train["Age"] = X_train["Age"].fillna(
        age_median
    )

    X_test["Age"] = X_test["Age"].fillna(
        age_median
    )

    # --------------------------------------
    # Handle missing Embarked
    # --------------------------------------

    embarked_mode = X_train["Embarked"].mode()[0]

    X_train["Embarked"] = X_train["Embarked"].fillna(
        embarked_mode
    )

    X_test["Embarked"] = X_test["Embarked"].fillna(
        embarked_mode
    )

    # --------------------------------------
    # Encode Sex
    # --------------------------------------

    X_train["Sex"] = X_train["Sex"].map({
        "male": 0,
        "female": 1
    })

    X_test["Sex"] = X_test["Sex"].map({
        "male": 0,
        "female": 1
    })

    # --------------------------------------
    # One-hot encode Embarked
    # --------------------------------------

    combined = pd.concat(
        [X_train, X_test],
        axis=0
    )

    combined = pd.get_dummies(
        combined,
        columns=["Embarked"],
        dtype=int
    )

    X_train = combined.iloc[
        :len(X_train)
    ].copy()

    X_test = combined.iloc[
        len(X_train):
    ].copy()

    return X_train, X_test


# ==========================================
# Prepare complete dataset
# ==========================================

def get_data(
    validation_size=0.2,
    random_state=42
):
    """
    Prepare training, validation and test data.

    Returns:
        X_train
        X_val
        X_test
        y_train
        y_val
        scaler
    """

    # --------------------------------------
    # Load datasets
    # --------------------------------------

    train_df, test_df = load_data()

    # --------------------------------------
    # Prepare features
    # --------------------------------------

    X, X_test = preprocess_features(
        train_df,
        test_df
    )

    # --------------------------------------
    # Prepare target
    # --------------------------------------

    y = train_df[TARGET].copy()

    # --------------------------------------
    # Split training data
    # into train + validation
    # --------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=validation_size,
        random_state=42,
        stratify=y
        
    )

    # --------------------------------------
    # Feature scaling
    # --------------------------------------

    scaler = StandardScaler()

    # IMPORTANT:
    # Fit scaler ONLY on training data

    X_train = scaler.fit_transform(
        X_train
    )

    # Use same scaler for validation

    X_val = scaler.transform(
        X_val
    )

    # Use same scaler for Kaggle test

    X_test = scaler.transform(
        X_test
    )

    # --------------------------------------
    # Convert features to tensors
    # --------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    # --------------------------------------
    # Convert targets to tensors
    # --------------------------------------

    y_train = torch.tensor(
        y_train.to_numpy(),
        dtype=torch.float32
    )

    y_val = torch.tensor(
        y_val.to_numpy(),
        dtype=torch.float32
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        scaler
    )