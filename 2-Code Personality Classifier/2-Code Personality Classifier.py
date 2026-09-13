# ============================================================
# CODE PERSONALITY CLASSIFIER
# Machine Learning Project
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "personality.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

try:
    dataset = pd.read_csv(
        DATASET_PATH,
        on_bad_lines="warn"
    )
except FileNotFoundError:
    print("\nERROR: personality.csv not found!")
    print("Make sure personality.csv is in the same folder as this Python file.")
    exit()


# ============================================================
# 3. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nFirst 5 Rows:")
print(dataset.head())

print("\nDataset Shape:")
print(dataset.shape)

print("\nDataset Info:")
dataset.info()

print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nDuplicate Rows:")
print(dataset.duplicated().sum())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "comment_count",
    "line_count",
    "blank_lines",
    "function_count",
    "avg_variable_length",
    "indentation_score",
    "loop_count",
    "personality"
]

missing_columns = [
    col for col in required_columns
    if col not in dataset.columns
]

if missing_columns:

    print("\nERROR: Missing columns:")
    print(missing_columns)

    print("\nExpected columns:")
    print(required_columns)

    exit()


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(dataset)

dataset = dataset.drop_duplicates().copy()

after_duplicates = len(dataset)

print("\nShape After Removing Duplicates:")
print(dataset.shape)

print(
    f"\nDuplicate rows removed: "
    f"{before_duplicates - after_duplicates}"
)


# ============================================================
# 6. DATA TYPE CLEANING
# ============================================================

print("\n" + "=" * 60)
print("DATA TYPE CLEANING")
print("=" * 60)


numeric_features = [
    "comment_count",
    "line_count",
    "blank_lines",
    "function_count",
    "avg_variable_length",
    "indentation_score",
    "loop_count"
]


# Convert feature columns to numeric
for column in numeric_features:

    dataset[column] = pd.to_numeric(
        dataset[column],
        errors="coerce"
    )


print("\nData Types After Conversion:")
print(dataset.dtypes)


# ============================================================
# 7. CHECK INVALID NUMERIC VALUES
# ============================================================

print("\nInvalid Numeric Values:")

invalid_values = dataset[numeric_features].isnull().sum()

print(invalid_values)


invalid_rows_before = len(dataset)

dataset = dataset.dropna(
    subset=numeric_features
).copy()

invalid_rows_after = len(dataset)

print(
    f"\nInvalid rows removed: "
    f"{invalid_rows_before - invalid_rows_after}"
)


# ============================================================
# 8. CONVERT NUMERIC COLUMNS
# ============================================================

dataset[numeric_features] = dataset[
    numeric_features
].astype(float)


# ============================================================
# 9. CHECK TARGET VALUES
# ============================================================

dataset["personality"] = dataset[
    "personality"
].astype(str).str.strip()


print("\nPersonality Classes:")
print(
    dataset["personality"].value_counts()
)


# ============================================================
# 10. DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATA QUALITY CHECK")
print("=" * 60)


# Check rows where blank lines are greater than line count
suspicious_rows = dataset[
    dataset["blank_lines"] > dataset["line_count"]
]

print(
    "\nRows where blank_lines > line_count:"
)

print(
    suspicious_rows[
        numeric_features + ["personality"]
    ]
)


# We will NOT automatically remove these rows.
# They may be noisy/synthetic data.


# ============================================================
# 11. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(
    dataset[numeric_features].describe()
)


# ============================================================
# 12. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# Personality Distribution
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.countplot(
    data=dataset,
    x="personality",
    order=dataset["personality"].value_counts().index
)

plt.title("Personality Class Distribution")

plt.xlabel("Personality")

plt.ylabel("Number of Samples")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "personality_distribution.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 13. FEATURE DISTRIBUTIONS
# ============================================================

for feature in numeric_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=dataset,
        x=feature,
        kde=True
    )

    plt.title(
        f"Distribution of {feature}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"{feature}_distribution.png"
        ),
        dpi=300
    )

    plt.show()


# ============================================================
# 14. CORRELATION HEATMAP
# ============================================================

plt.figure(
    figsize=(11, 8)
)

correlation = dataset[
    numeric_features
].corr()

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Feature Correlation Heatmap"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 15. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)


# ------------------------------------------------------------
# Comment Ratio
# ------------------------------------------------------------

dataset["comment_ratio"] = (
    dataset["comment_count"]
    /
    dataset["line_count"].clip(lower=1)
)


# ------------------------------------------------------------
# Function Density
# ------------------------------------------------------------

dataset["function_density"] = (
    dataset["function_count"]
    /
    dataset["line_count"].clip(lower=1)
)


# ------------------------------------------------------------
# Loop Density
# ------------------------------------------------------------

dataset["loop_density"] = (
    dataset["loop_count"]
    /
    dataset["line_count"].clip(lower=1)
)


# ------------------------------------------------------------
# Blank Line Ratio
# ------------------------------------------------------------

dataset["blank_line_ratio"] = (
    dataset["blank_lines"]
    /
    dataset["line_count"].clip(lower=1)
)


# ------------------------------------------------------------
# Complexity Score
# ------------------------------------------------------------

dataset["complexity_score"] = (
    dataset["line_count"]
    +
    (2 * dataset["function_count"])
    +
    (2 * dataset["loop_count"])
    +
    dataset["blank_lines"]
)


# ------------------------------------------------------------
# Code Structure Score
# ------------------------------------------------------------

dataset["code_structure_score"] = (
    dataset["indentation_score"]
    +
    dataset["function_count"]
    +
    dataset["loop_count"]
)


print("\nNew Features Created:")

new_features = [
    "comment_ratio",
    "function_density",
    "loop_density",
    "blank_line_ratio",
    "complexity_score",
    "code_structure_score"
]

print(new_features)


# ============================================================
# 16. DEFINE FEATURES AND TARGET
# ============================================================

all_features = numeric_features + new_features

X = dataset[all_features].copy()

y = dataset["personality"].copy()


print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)


# ============================================================
# 17. LABEL ENCODING
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print("\nPersonality Encoding:")

for number, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{number} = {label}"
    )


# ============================================================
# 18. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)


print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

print(
    f"\nTraining Samples: {len(X_train)}"
)

print(
    f"Testing Samples: {len(X_test)}"
)


# ============================================================
# 19. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {

    "Logistic Regression": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ]),


    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),


    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),


    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    ),


    "SVM": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            SVC(
                kernel="rbf"
            )
        )
    ])
}


# ============================================================
# 20. TRAIN AND EVALUATE MODELS
# ============================================================

print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)


results = []

trained_models = {}


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    trained_models[name] = model


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )


    # --------------------------------------------------------
    # Cross Validation
    # --------------------------------------------------------

    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1_weighted"
    )

    cv_f1 = cv_scores.mean()


    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1 Score": f1,

        "CV F1 Score": cv_f1
    })


    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"CV F1    : {cv_f1:.4f}"
    )


# ============================================================
# 21. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="CV F1 Score",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# Save results
results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# ============================================================
# 22. MODEL COMPARISON GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

x_positions = np.arange(
    len(results_df)
)

width = 0.2


plt.bar(
    x_positions - width * 1.5,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x_positions - width / 2,
    results_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x_positions + width / 2,
    results_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x_positions + width * 1.5,
    results_df["F1 Score"],
    width,
    label="F1 Score"
)


plt.xticks(
    x_positions,
    results_df["Model"],
    rotation=20
)

plt.ylabel("Score")

plt.title(
    "Machine Learning Model Comparison"
)

plt.ylim(
    0,
    1.05
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 23. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[
    best_model_name
]


print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"CV F1 Score: "
    f"{results_df.iloc[0]['CV F1 Score']:.4f}"
)


# ============================================================
# 24. BEST MODEL PREDICTION
# ============================================================

best_predictions = best_model.predict(
    X_test
)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        best_predictions,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ============================================================
# 26. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    best_predictions
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted Personality"
)

plt.ylabel(
    "Actual Personality"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.show()


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)


feature_importance = None


# Random Forest / Decision Tree / Gradient Boosting
if hasattr(
    best_model,
    "feature_importances_"
):

    feature_importance = (
        best_model.feature_importances_
    )


# Pipeline models
elif hasattr(
    best_model,
    "named_steps"
):

    final_model = best_model.named_steps[
        "model"
    ]


    if hasattr(
        final_model,
        "coef_"
    ):

        feature_importance = np.mean(
            np.abs(
                final_model.coef_
            ),
            axis=0
        )


if feature_importance is not None:

    importance_df = pd.DataFrame({

        "Feature": all_features,

        "Importance": feature_importance

    })


    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )


    print(
        importance_df.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Feature Importance Graph
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 7)
    )

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Feature Importance - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "feature_importance.png"
        ),
        dpi=300
    )

    plt.show()


else:

    print(
        "\nFeature importance is not directly available "
        "for this model."
    )


# ============================================================
# 28. PERSONALITY PREDICTION FUNCTION
# ============================================================

def predict_personality(
    comment_count,
    line_count,
    blank_lines,
    function_count,
    avg_variable_length,
    indentation_score,
    loop_count
):

    # --------------------------------------------------------
    # Create input dictionary
    # --------------------------------------------------------

    sample = pd.DataFrame({

        "comment_count": [
            comment_count
        ],

        "line_count": [
            line_count
        ],

        "blank_lines": [
            blank_lines
        ],

        "function_count": [
            function_count
        ],

        "avg_variable_length": [
            avg_variable_length
        ],

        "indentation_score": [
            indentation_score
        ],

        "loop_count": [
            loop_count
        ]
    })


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    sample["comment_ratio"] = (
        sample["comment_count"]
        /
        sample["line_count"].clip(lower=1)
    )


    sample["function_density"] = (
        sample["function_count"]
        /
        sample["line_count"].clip(lower=1)
    )


    sample["loop_density"] = (
        sample["loop_count"]
        /
        sample["line_count"].clip(lower=1)
    )


    sample["blank_line_ratio"] = (
        sample["blank_lines"]
        /
        sample["line_count"].clip(lower=1)
    )


    sample["complexity_score"] = (
        sample["line_count"]
        +
        (2 * sample["function_count"])
        +
        (2 * sample["loop_count"])
        +
        sample["blank_lines"]
    )


    sample["code_structure_score"] = (
        sample["indentation_score"]
        +
        sample["function_count"]
        +
        sample["loop_count"]
    )


    # --------------------------------------------------------
    # Arrange columns exactly like training data
    # --------------------------------------------------------

    sample = sample[
        all_features
    ]


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = best_model.predict(
        sample
    )


    personality = label_encoder.inverse_transform(
        prediction
    )[0]


    return personality


# ============================================================
# 29. TEST NEW CODE SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("TESTING NEW CODE SAMPLE")
print("=" * 60)


sample_personality = predict_personality(

    comment_count=1,

    line_count=10,

    blank_lines=1,

    function_count=2,

    avg_variable_length=6.0,

    indentation_score=7,

    loop_count=1
)


print(
    "\nPredicted Personality:"
)

print(
    sample_personality
)


# ============================================================
# 30. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED")
print("=" * 60)

print(
    f"\nFinal Dataset Shape: {dataset.shape}"
)

print(
    f"Number of Features: {len(all_features)}"
)

print(
    f"Number of Personality Classes: "
    f"{len(label_encoder.classes_)}"
)

print(
    f"Best Model: {best_model_name}"
)

print(
    "\nOutput files saved in:"
)

print(
    OUTPUT_DIR
)

print("\nGenerated files include:")

print(
    "- personality_distribution.png"
)

print(
    "- feature distribution graphs"
)

print(
    "- correlation_heatmap.png"
)

print(
    "- model_comparison.png"
)

print(
    "- model_comparison.csv"
)

print(
    "- confusion_matrix.png"
)

print(
    "- feature_importance.png"
)

print("\n" + "=" * 60)