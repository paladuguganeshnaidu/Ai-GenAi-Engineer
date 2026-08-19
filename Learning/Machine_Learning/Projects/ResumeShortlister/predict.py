from pathlib import Path
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"
feature_columns = joblib.load(MODELS_DIR / "feature_columns.pkl")
model = joblib.load(MODELS_DIR / "resume_model.pkl")
degree_encoder = joblib.load(MODELS_DIR / "degree_encoder.pkl")
target_encoder = joblib.load(MODELS_DIR / "target_encoder.pkl")

SKILLS = [
    "Python",
    "Java",
    "C++",
    "SQL",
    "TensorFlow",
    "PyTorch",
    "Docker",
    "Git",
    "Linux"
]

FIELDS = {
    "CollegeTier": int,
    "Experience": int,
    "DSA": int,
    "Communication": int,
    "ProblemSolving": int,
    "Internships": int,
    "Certifications": int,
    "Hackathons": int,
    "Projects": int,
    "CGPA": float,
    "Aptitude": int,
    "InterviewScore": int,
    "LeetCodeSolved": int,
    "ExpectedSalaryLPA": float,
    "NoticePeriodDays": int
}

candidate = {}

degree = input("Degree: ").strip().lower()

try:
    candidate["Degree"] = degree_encoder.transform([degree])[0]
except ValueError:
    print(f"❌ '{degree}' is not a valid Degree.")
    print("Valid Degrees:", list(degree_encoder.classes_))
    exit()


for field, dtype in FIELDS.items():
    while True:
        try:
            value = dtype(input(f"{field}: "))
            candidate[field] = value
            break
        except ValueError:
            print(f"❌ Please enter a valid {dtype.__name__} value.")

for skill in SKILLS:
    while True:
        value = input(f"{skill} (1/0): ").strip()

        if value in ("0", "1"):
            candidate[skill] = int(value)
            break

        print("❌ Enter only 0 or 1.")

candidate = pd.DataFrame([candidate])
candidate = candidate[feature_columns]
prediction = model.predict(candidate)
probability = model.predict_proba(candidate)

result = target_encoder.inverse_transform(prediction)

confidence = probability.max() * 100
prediction = target_encoder.inverse_transform(prediction)[0]

if prediction == "yes":
    result = "Eligible for this Job"
else:
    result = "Not Eligible for this Job"

print("\n==========================")
print("Resume Evaluation")
print("==========================")
print("Status      :", result)
print(f"Confidence  : {confidence:.2f}%")