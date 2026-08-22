import pandas as pd
import joblib
from pathlib import Path

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parents[1]

MODELS_DIR = BASE_DIR / "Models"
DATA_DIR = REPO_DIR / "Data" / "ResumeShortlister" / "Training"
TEST_DIR = REPO_DIR / "Data" / "ResumeShortlister" / "Testing"
TEXT_RE = BASE_DIR / "Tests Results"
RESULTS_DIR = BASE_DIR / "Results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# Load Model & Encoders
# =====================================================
model = joblib.load(MODELS_DIR / "resume_model.pkl")
degree_encoder = joblib.load(MODELS_DIR / "degree_encoder.pkl")
branch_encoder = joblib.load(MODELS_DIR / "branch_encoder.pkl")
target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")

# =====================================================
# Load CSV
# =====================================================
input_file = TEST_DIR / "test_candidates.csv"

df = pd.read_csv(input_file)

original_df = df.copy()

# =====================================================
# Preprocessing
# =====================================================
df["Degree"] = df["Degree"].str.strip().str.lower()

df["Degree"] = degree_encoder.transform(df["Degree"])

# Remove Name before prediction
X = df.drop(columns=["Name"])

# Keep feature order exactly same as training
X = X[feature_columns]

# =====================================================
# Prediction
# =====================================================
predictions = model.predict(X)
probabilities = model.predict_proba(X)

status_list = []
confidence_list = []

for pred, prob in zip(predictions, probabilities):

    label = target_encoder.inverse_transform([pred])[0]

    confidence = max(prob) * 100

    if label == "yes":
        status = "Eligible for this Job"
    else:
        status = "Not Eligible for this Job"

    status_list.append(status)
    confidence_list.append(round(confidence, 2))

# =====================================================
# Save Results
# =====================================================
original_df["Status"] = status_list
original_df["Confidence (%)"] = confidence_list

output_file = TEXT_RE / "predictions.csv"

original_df.to_csv(output_file, index=False)

print("=" * 50)
print(" Resume Screening Completed")
print("=" * 50)
print(f"Candidates Processed : {len(original_df)}")
print(f"Results Saved To     : {output_file}")
print("=" * 50)