# Resume Shortlister using Machine Learning

An AI-powered Resume Shortlister that predicts whether a candidate is **Eligible** or **Not Eligible** for a job based on their education, technical skills, experience, academic performance, and project portfolio.

The project demonstrates the complete Machine Learning workflow—from dataset preparation and model training to prediction and batch evaluation—using **Python** and **Scikit-learn**.

---

## Project Overview

Recruiters often spend hours manually reviewing resumes. This project automates the initial screening process by training a Machine Learning model on candidate data and predicting whether new applicants meet the desired requirements.

The system supports:

* Training a Machine Learning model
* Predicting a single candidate
* Screening thousands of resumes from a CSV file
* Exporting results into a new CSV with prediction status and confidence score

---

# Features

* Machine Learning based resume screening
* Decision Tree Classification
* Single resume prediction
* Batch prediction from CSV
* Confidence score for every prediction
* Automatic CSV result generation
* Label Encoding for categorical data
* Model serialization using Joblib
* Easy to extend with new datasets

---

# Technologies Used

* Python 3
* Pandas
* Scikit-learn
* Joblib

---

# Machine Learning Algorithm

Current Model:

* Decision Tree Classifier

Preprocessing:

* Label Encoding
* Feature Selection

Model Persistence:

* Joblib (.pkl)

---

# Dataset

## Training Dataset

File:

```text
../../../../Data/ResumeShortlister/Training/candidates.csv
```

Purpose:

Used for training the Decision Tree model.

Current Size:

**100,000 candidate records**

Each record contains candidate information such as:

* Name
* Degree
* Branch
* College Tier
* Experience
* Python
* Java
* C++
* SQL
* TensorFlow
* PyTorch
* Docker
* Git
* Linux
* Projects
* CGPA
* Certifications
* Internship
* Communication
* Aptitude
* Selected (Target)

---

## Testing Dataset

File:

```text
../../../../Data/ResumeShortlister/Testing/test_candidates.csv
```

Purpose:

Used to evaluate the trained model.

Current Size:

**10,000 candidate records**

This dataset contains unseen candidate profiles that are passed through the trained model.

---

# Output Dataset

File:

```text
Tests Results/predictions.csv
```

Generated automatically after batch prediction.

Additional columns added:

* Status
* Confidence (%)

Example:

| Name   | Status                    | Confidence |
| ------ | ------------------------- | ---------- |
| Ganesh | Eligible for this Job     | 98.74%     |
| Rahul  | Not Eligible for this Job | 100.00%    |

---

# Project Structure

```text
Learning/Machine_Learning/Projects/ResumeShortlister/

│
├── Models/
│   ├── resume_model.pkl
│   ├── degree_encoder.pkl
│   ├── branch_encoder.pkl
│   ├── target_encoder.pkl
│   └── feature_columns.pkl
│
├── Results/
│   └── (training outputs)
│
├── Tests Results/
│   └── predictions.csv
│
├── train.py
├── predict.py
├── batch_predict.py
│
└── README.md
```

---

# Workflow

```text
Training Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Label Encoding
        │
        ▼
Feature Extraction
        │
        ▼
Decision Tree Training
        │
        ▼
Save Model (.pkl)
        │
──────────────
        │
        ▼
Load Trained Model
        │
        ▼
Input Test CSV
        │
        ▼
Predict Each Candidate
        │
        ▼
Generate Confidence Score
        │
        ▼
Create predictions.csv
```

---

# Model Training

Run:

```bash
python train.py
```

This will:

* Load the training dataset
* Encode categorical values
* Train the Decision Tree model
* Save the trained model
* Save all encoders
* Save feature order

Generated files:

* resume_model.pkl
* degree_encoder.pkl
* branch_encoder.pkl
* target_encoder.pkl
* feature_columns.pkl

---

# Single Candidate Prediction

Run:

```bash
python predict.py
```

Example Output

```text
==========================
Resume Evaluation
==========================

Status      : Eligible for this Job
Confidence  : 98.52%
```

---

# Batch Prediction

Run:

```bash
python batch_predict.py
```

Input:

```text
Data/test_candidates.csv
```

Output:

```text
Results/predictions.csv
```

---

# Current Results

Training Samples:

**100,000**

Testing Samples:

**10,000**

Prediction Results:

* Eligible for this Job: **5,525**
* Not Eligible for this Job: **4,475**

Every candidate receives:

* Prediction Status
* Confidence Score

---

# Confidence Score

The confidence score represents the model's estimated probability for the predicted class.

Example:

```text
Status      : Eligible for this Job
Confidence  : 97.84%
```

A higher confidence indicates the model is more certain about its prediction based on the learned patterns from the training data.

---

# Learning Objectives

This project demonstrates practical Machine Learning concepts including:

* Data Cleaning
* Data Preprocessing
* Feature Engineering
* Label Encoding
* Decision Tree Classification
* Model Training
* Model Serialization
* Batch Prediction
* Confidence Estimation
* CSV Data Processing

---

# Future Improvements

* Random Forest Classifier
* XGBoost
* LightGBM
* CatBoost
* Resume PDF Parsing
* NLP-based Skill Extraction
* Explainable AI (Feature Importance)
* Web Interface using Flask or FastAPI
* REST API Integration
* Recruiter Dashboard
* Candidate Ranking
* Multi-job Recommendation
* Interview Score Prediction
* Resume Matching using LLMs
* ATS Score Generation

---

# Repository

Source Code:

https://github.com/paladuguganeshnaidu/Ai-GenAi-Engineer/tree/main/Learning/Machine_Learning/Projects/ResumeShortlister

---

# Author

**Ganesh Paladugu Naidu**

AI Engineering Student | Machine Learning | Generative AI | Python | Scikit-learn

This project was built as part of my AI Engineering learning journey to understand end-to-end Machine Learning workflows, from data preprocessing and model training to automated resume screening and batch prediction.
