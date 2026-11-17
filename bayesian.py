
import pandas as pd
import numpy as np

from ucimlrepo import fetch_ucirepo

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.inference import VariableElimination


# --------------------------------------------------
# STEP 1: LOAD CLEVELAND HEART DISEASE DATASET
# --------------------------------------------------

print("=" * 60)
print("       BAYESIAN NETWORK - HEART DISEASE DIAGNOSIS")
print("=" * 60)

print("\nLoading Cleveland Heart Disease Dataset...")

heart = fetch_ucirepo(id=45)

data = pd.concat(
    [heart.data.features, heart.data.targets],
    axis=1
)

print("\nFirst five records:")
print(data.head())


# --------------------------------------------------
# STEP 2: DATA PREPROCESSING
# --------------------------------------------------

print("\nPreprocessing dataset...")

data = data.replace("?", np.nan)

data.columns = [
    str(col).strip().lower()
    for col in data.columns
]

# Rename target column
data.rename(
    columns={"num": "heartdisease"},
    inplace=True
)

# Convert all columns to numeric
for column in data.columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

# Remove missing records
data.dropna(inplace=True)

# Discretize continuous attributes

data["age"] = pd.cut(
    data["age"],
    bins=[0, 40, 50, 60, 100],
    labels=["young", "middle", "senior", "old"]
)

data["trestbps"] = pd.cut(
    data["trestbps"],
    bins=[0, 120, 140, 300],
    labels=["low", "normal", "high"]
)

data["chol"] = pd.cut(
    data["chol"],
    bins=[0, 200, 240, 600],
    labels=["normal", "borderline", "high"]
)

data["thalach"] = pd.cut(
    data["thalach"],
    bins=[0, 100, 150, 250],
    labels=["low", "medium", "high"]
)

# Select required attributes
selected_columns = [
    "age",
    "sex",
    "trestbps",
    "fbs",
    "exang",
    "chol",
    "thalach",
    "heartdisease"
]

data = data[selected_columns].dropna().copy()

# Convert categorical columns to strings
for column in data.columns:
    if column in ["age", "trestbps", "chol", "thalach"]:
        data[column] = data[column].astype(str)
    else:
        data[column] = data[column].astype(int).astype(str)

print("\nPreprocessed Dataset:")
print(data.head())

print("\nTotal Records:", len(data))


# --------------------------------------------------
# STEP 3: CONSTRUCT BAYESIAN NETWORK
# --------------------------------------------------

print("\nConstructing Bayesian Network...")

model = DiscreteBayesianNetwork([
    ("age", "trestbps"),
    ("age", "fbs"),
    ("sex", "trestbps"),
    ("exang", "trestbps"),
    ("trestbps", "heartdisease"),
    ("fbs", "heartdisease"),
    ("heartdisease", "thalach"),
    ("heartdisease", "chol")
])

print("\nBayesian Network Structure:")

print(model.edges())


# --------------------------------------------------
# STEP 4: LEARN CONDITIONAL PROBABILITY DISTRIBUTIONS
# --------------------------------------------------

print("\nLearning Conditional Probability Distributions...")

model.fit(
    data,
    estimator=DiscreteMLE()
)

print("\nCPDs Successfully Learned!")

for cpd in model.get_cpds():
    print("\n", cpd)


# --------------------------------------------------
# STEP 5: PERFORM INFERENCE
# --------------------------------------------------

print("\n" + "=" * 60)
print("             HEART DISEASE INFERENCE")
print("=" * 60)

inference = VariableElimination(model)


# Query 1: Heart disease probability given age

print("\n1. Probability of Heart Disease given Age = 28")

result1 = inference.query(
    variables=["heartdisease"],
    evidence={"age": "young"}
)

print(result1)


# Query 2: Heart disease probability given cholesterol

print("\n2. Probability of Heart Disease given High Cholesterol")

result2 = inference.query(
    variables=["heartdisease"],
    evidence={"chol": "high"}
)

print(result2)


# --------------------------------------------------
# STEP 6: DISPLAY FINAL DIAGNOSIS
# --------------------------------------------------

print("\n" + "=" * 60)
print("                 FINAL RESULTS")
print("=" * 60)

print("\nDiagnosis based on Age = 28:")

for i, probability in enumerate(result1.values):
    print(
        f"Heart Disease Level {i}: "
        f"{probability:.4f}"
    )

print("\nDiagnosis based on High Cholesterol:")

for i, probability in enumerate(result2.values):
    print(
        f"Heart Disease Level {i}: "
        f"{probability:.4f}"
    )

print("\nProgram Executed Successfully!")

print("=" * 60)
