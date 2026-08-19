import time
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[3]
DATA_DIR = REPO_DIR / "Data" / "ResumeShortlister" / "Training"
MODELS_DIR = BASE_DIR / "Models"

overall_start = time.perf_counter()
print("Data Loading & Encoding Started........")
start = time.perf_counter()

df = pd.read_csv(DATA_DIR / "candidates.csv")

df["Degree"] = df["Degree"].astype(str).str.strip().str.lower()
df["Selected"] = df["Selected"].astype(str).str.strip().str.lower()

degree_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["Degree"] = degree_encoder.fit_transform(df["Degree"])
df["Selected"] = target_encoder.fit_transform(df["Selected"])

end = time.perf_counter()

print(f"Data Loading & Encoding : {end - start:.2f} seconds")

start = time.perf_counter()

X = df.drop(columns=["Name", "Selected"])
y = df["Selected"]

joblib.dump(list(X.columns), MODELS_DIR / "feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

end = time.perf_counter()


print(f"Feature Preparation    : {end - start:.2f} seconds")
print(f"Training Samples       : {len(X_train):,}")
print(f"Testing Samples        : {len(X_test):,}")

print("============================================================")
print("Resume Shortlister - Decision Tree Training")
print("============================================================")

start = time.perf_counter()

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

end = time.perf_counter()

print(f"Training Time          : {end - start:.2f} seconds")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
precision=precision_score(y_test,predictions)
recall=recall_score(y_test,predictions)
f1=f1_score(y_test,predictions)
report=classification_report(y_test,predictions)

print("Confusion Matrix")
print(cm)
print(f"Accuracy               : {accuracy * 100:.2f}%")

start = time.perf_counter()

joblib.dump(model, MODELS_DIR / "resume_model.pkl")
joblib.dump(degree_encoder, MODELS_DIR / "degree_encoder.pkl")
joblib.dump(target_encoder, MODELS_DIR / "target_encoder.pkl")

end = time.perf_counter()

print(f"Model Saving Time      : {end - start:.2f} seconds")

overall_end = time.perf_counter()

print(f"Total Training Time    : {overall_end - overall_start:.2f} seconds")
print("============================================================")
print("Model Trained Successfully")
print("============================================================")
print("======================================================")
print("Model Evaluation")
print("======================================================")

print(f"Accuracy              : {accuracy * 100:.2f}%")
print(f"Precision             : {precision * 100:.2f}%")
print(f"Recall                : {recall * 100:.2f}%")
print(f"F1 Score              : {f1 * 100:.2f}%")

print("Confusion Matrix")
print(cm)

print("Classification Report")
print(report)