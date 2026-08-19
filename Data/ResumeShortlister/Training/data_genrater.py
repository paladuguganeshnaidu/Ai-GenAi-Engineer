import csv
import random
import math
import os
import signal
import time
from pathlib import Path
from tqdm import tqdm

# ==========================================================
# CONFIGURATION
# ==========================================================

OUTPUT_FILE = Path(__file__).resolve().parent / "candidates.csv"

TOTAL_ROWS = 10_000_000

RANDOM_SEED = 2026

random.seed(RANDOM_SEED)

# ==========================================================
# DEGREES
# ==========================================================

DEGREES = [
    "BTech",
    "BE",
    "BCA",
    "MCA",
    "MTech",
    "MSc"
]

# Degree probabilities

DEGREE_WEIGHTS = [
    0.36,
    0.20,
    0.12,
    0.12,
    0.10,
    0.10
]

# ==========================================================
# FIRST NAMES
# ==========================================================

FIRST_NAMES = [

"Aarav","Vivaan","Aditya","Vihaan","Arjun","Sai","Rahul","Ganesh","Kiran","Rohit",
"Abhishek","Akash","Harsha","Surya","Varun","Nikhil","Siddharth","Manoj","Rakesh",
"Deepak","Yash","Krishna","Tejas","Ashwin","Vivek","Farhan","Imran","Suman",
"Anjali","Priya","Sneha","Pooja","Kavya","Aditi","Nisha","Meera","Divya","Shreya",
"Riya","Neha","Ananya","Ishita","Keerthi","Lakshmi","Bhavana","Swathi","Madhavi",
"Sanjana","Reshma","Aisha",

"Harsh","Pranav","Mohan","Sathvik","Chandan","Vikas","Ajay","Aravind","Praveen",
"Lokesh","Madhu","Naveen","Mahesh","Tarun","Vignesh","Dinesh","Ramesh","Karthik",
"Darshan","Nitin","Bhargav","Roshan","Srinivas","Vinay","Bharath","Kishore",
"Raghu","Prakash","Gautham","Pavan","Venu","Shyam","Rohan","Aman","Aryan","Kabir",
"Dev","Parth","Dhruv","Rudra","Shaurya","Om","Ayaan","Reyansh","Ved","Atharv",
"Ishaan","Nakul","Prithvi","Abhay"

]

# ==========================================================
# LAST NAMES
# ==========================================================

LAST_NAMES = [

"Sharma","Naidu","Reddy","Patel","Verma","Singh","Kumar","Gupta","Joshi","Iyer",
"Nair","Rao","Das","Mishra","Kulkarni","Jain","Yadav","Shetty","Gowda","Menon",
"Malhotra","Bhat","Saxena","Chauhan","Chowdhury","Pandey","Shukla","Thakur",
"Dubey","Tripathi","Soni","Bansal","Khanna","Kapoor","Aggarwal","Mehta",
"Bhattacharya","Mukherjee","Pal","Pillai","Fernandes","D'Souza","Paul","Thomas",
"George","Mathew","Khan","Ali","Ansari","Shaikh"

]

# ==========================================================
# CSV HEADER
# ==========================================================

HEADER = [

"Name",
"Degree",
"Experience",
"Python",
"Java",
"C++",
"SQL",
"TensorFlow",
"PyTorch",
"Docker",
"Git",
"Linux",
"Projects",
"CGPA",
"Selected"

]

# ==========================================================
# NAME GENERATOR
# ==========================================================

def generate_name(index):

    return (
        random.choice(FIRST_NAMES)
        + " "
        + random.choice(LAST_NAMES)
        + " "
        + str(index)
    )

# ==========================================================
# DEGREE
# ==========================================================

def generate_degree():

    return random.choices(

        DEGREES,

        weights=DEGREE_WEIGHTS,

        k=1

    )[0]

# ==========================================================
# EXPERIENCE
# ==========================================================

def generate_experience():

    r = random.random()

    if r < 0.22:
        return 0

    elif r < 0.40:
        return 1

    elif r < 0.57:
        return 2

    elif r < 0.72:
        return 3

    elif r < 0.84:
        return 4

    elif r < 0.92:
        return 5

    elif r < 0.97:
        return 6

    elif r < 0.99:
        return 7

    else:
        return 8

# ==========================================================
# CGPA
# ==========================================================

def generate_cgpa(exp):

    mean = 7.8 - exp * 0.05

    cgpa = random.gauss(mean, 0.8)

    return round(

        max(

            5.5,

            min(

                9.9,

                cgpa

            )

        ),

        2

    )
# ==========================================================
# SKILL GENERATION
# ==========================================================

def generate_skills(exp, degree):

    if degree in ("MTech", "MSc"):
        ai_bonus = 0.10
    elif degree == "MCA":
        ai_bonus = 0.06
    else:
        ai_bonus = 0.00

    python = int(random.random() < (0.40 + exp * 0.05 + ai_bonus))

    java = int(random.random() < (0.45 + exp * 0.03))

    cpp = int(random.random() < (0.42 + exp * 0.02))

    sql = int(random.random() < (0.35 + exp * 0.05))

    git = int(random.random() < (0.45 + exp * 0.04))

    linux = int(random.random() < (0.32 + exp * 0.05))

    docker = int(random.random() < (0.18 + exp * 0.05))

    tensorflow = 0
    pytorch = 0

    if python:

        tensorflow = int(
            random.random() <
            (0.10 + exp * 0.05 + ai_bonus)
        )

        pytorch = int(
            random.random() <
            (0.10 + exp * 0.05 + ai_bonus)
        )

    # impossible combinations (rare)

    if not python and random.random() < 0.98:
        tensorflow = 0
        pytorch = 0

    return (

        python,

        java,

        cpp,

        sql,

        tensorflow,

        pytorch,

        docker,

        git,

        linux

    )


# ==========================================================
# PROJECTS
# ==========================================================

def generate_projects(exp):

    base = int(random.gauss(2 + exp * 0.7, 2))

    return max(0, min(base, 10))


# ==========================================================
# HIDDEN HIRING SCORE
# ==========================================================

def calculate_score(

    degree,

    exp,

    python,

    java,

    cpp,

    sql,

    tensorflow,

    pytorch,

    docker,

    git,

    linux,

    projects,

    cgpa

):

    score = 0.0

    # ------------------------------------
    # Technical Stack
    # ------------------------------------

    if python:
        score += 2

    if sql:
        score += 1.5

    if git:
        score += 1.2

    if docker:
        score += 1.4

    if linux:
        score += 1

    if tensorflow:
        score += 1.8

    if pytorch:
        score += 1.8

    if java:
        score += 0.7

    if cpp:
        score += 0.6

    # ------------------------------------
    # Hidden interactions
    # ------------------------------------

    if python and sql and git:
        score += 2.5

    if docker and linux and git:
        score += 2.2

    if tensorflow and pytorch:
        score += 2.5

    if java and cpp:
        score += 1.2

    if exp >= 4 and projects >= 5:
        score += 3.5

    if exp <= 1 and projects >= 8:
        score += 2.8

    if cgpa < 7.0 and projects >= 8:
        score += 2.5

    if cgpa > 9.2 and projects <= 1:
        score -= 3

    if exp >= 7 and projects <= 2:
        score -= 2

    # ------------------------------------
    # Degree effects
    # ------------------------------------

    if degree == "MTech":
        score += 0.8

    elif degree == "MSc":
        score += 0.5

    elif degree == "BCA":
        score -= 0.2

    # ------------------------------------
    # Projects
    # ------------------------------------

    score += projects * 0.45

    # ------------------------------------
    # Experience
    # ------------------------------------

    score += exp * 0.35

    # ------------------------------------
    # CGPA
    # ------------------------------------

    score += (cgpa - 7.5) * 0.45

    # ------------------------------------
    # Random recruiter behaviour
    # ------------------------------------

    score += random.gauss(0, 1.2)

    # rare lucky candidate

    if random.random() < 0.02:
        score += 4

    # rare rejection

    if random.random() < 0.03:
        score -= 4

    return score


# ==========================================================
# FINAL LABEL
# ==========================================================

def generate_label(score):

    if score >= 8.3:

        result = "Yes"

    else:

        result = "No"

    # recruiter disagreement

    if random.random() < 0.12:

        result = "No" if result == "Yes" else "Yes"

    return result


# ==========================================================
# BUILD ONE CANDIDATE
# ==========================================================

def generate_candidate(index):

    name = generate_name(index)

    degree = generate_degree()

    experience = generate_experience()

    cgpa = generate_cgpa(experience)

    (
        python,
        java,
        cpp,
        sql,
        tensorflow,
        pytorch,
        docker,
        git,
        linux

    ) = generate_skills(

        experience,

        degree

    )

    projects = generate_projects(experience)

    score = calculate_score(

        degree,

        experience,

        python,

        java,

        cpp,

        sql,

        tensorflow,

        pytorch,

        docker,

        git,

        linux,

        projects,

        cgpa

    )

    selected = generate_label(score)

    return [

        name,

        degree,

        experience,

        python,

        java,

        cpp,

        sql,

        tensorflow,

        pytorch,

        docker,

        git,

        linux,

        projects,

        cgpa,

        selected

    ]
# ==========================================================
# CSV WRITER
# ==========================================================

def write_dataset():

    start_time = time.time()

    print("=" * 60)
    print(" Resume Shortlisting Dataset Generator")
    print("=" * 60)
    print(f"Output File : {OUTPUT_FILE}")
    print(f"Rows        : {TOTAL_ROWS:,}")
    print()

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8",
        buffering=1024 * 1024
    ) as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(HEADER)

        for i in tqdm(

            range(1, TOTAL_ROWS + 1),

            desc="Generating Candidates",

            unit=" rows"

        ):

            writer.writerow(

                generate_candidate(i)

            )

            # Flush every 100000 rows

            if i % 100000 == 0:

                csvfile.flush()

    end_time = time.time()

    elapsed = end_time - start_time

    print()
    print("=" * 60)
    print("Dataset Generated Successfully")
    print("=" * 60)
    print(f"Rows Generated : {TOTAL_ROWS:,}")
    print(f"Saved As       : {OUTPUT_FILE}")
    print(f"Time Taken     : {elapsed/60:.2f} Minutes")
    print("=" * 60)


# ==========================================================
# CTRL + C HANDLER
# ==========================================================

def signal_handler(sig, frame):

    print()

    print("Generation Interrupted.")

    print("Current CSV has been saved.")

    exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    write_dataset()