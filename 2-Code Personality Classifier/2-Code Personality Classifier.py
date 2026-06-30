from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# GLOBAL STYLING
# ==================================================

sns.set_theme(style="darkgrid", palette="Set2")
plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 12
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13

# ==================================================
# LOAD DATA
# ==================================================
BASE_DIR = Path(__file__).parent

IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)

dataset = pd.read_csv(BASE_DIR / "personality.csv")

print("\nFirst 5 Rows")
print(dataset.head())

print("\nDataset Info")
print(dataset.info())

print("\nStatistical Summary")
print(dataset.describe())

# Remove duplicate rows
dataset.drop_duplicates(inplace=True)

print("\nData Types")
print(dataset.dtypes)

# ==================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================

# 1. Personality Distribution
plt.figure(figsize=(8, 5))

sns.countplot(
    x="personality",
    data=dataset,
    hue="personality",
    palette="viridis",
    legend=False
)

plt.title("Distribution of Personality Types")
plt.xlabel("Personality")
plt.ylabel("Count")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(IMAGE_DIR / "personality_distribution.png", dpi=300, bbox_inches="tight")
plt.show()



# 2. Line Count vs Personality
plt.figure(figsize=(9, 5))

sns.boxplot(
    x="personality",
    y="line_count",
    data=dataset,
    palette="Set3"
)

plt.title("Line Count by Personality Type")
plt.xlabel("Personality")
plt.ylabel("Line Count")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(IMAGE_DIR / "line_count.png", dpi=300, bbox_inches="tight")
plt.show()



# 3. Function Count vs Personality
plt.figure(figsize=(9, 5))

sns.boxplot(
    x="personality",
    y="function_count",
    data=dataset,
    palette="coolwarm"
)

plt.title("Function Count by Personality Type")
plt.xlabel("Personality")
plt.ylabel("Function Count")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(IMAGE_DIR / "function_count.png", dpi=300, bbox_inches="tight")
plt.show()



# 4. Comment Count vs Personality
plt.figure(figsize=(9, 5))

sns.boxplot(
    x="personality",
    y="comment_count",
    data=dataset,
    palette="rocket"
)

plt.title("Comment Count by Personality Type")
plt.xlabel("Personality")
plt.ylabel("Comment Count")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(IMAGE_DIR / "comment_count.png", dpi=300, bbox_inches="tight")
plt.show()



# 5. Correlation Heatmap
plt.figure(figsize=(10, 7))

corr = dataset.corr(numeric_only=True)

sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=1,
    square=True
)

plt.title("Feature Correlation Heatmap")
plt.tight_layout()

plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# FEATURE ENGINEERING
# ==================================================

dataset["complexity_score"] = (
    dataset["line_count"]
    + dataset["function_count"] * 2
    + dataset["loop_count"] * 2
    + dataset["comment_count"]
)

# ==================================================
# LABEL ENCODING
# ==================================================

le = LabelEncoder()

dataset["personality"] = le.fit_transform(
    dataset["personality"]
)

# ==================================================
# TRAIN-TEST SPLIT
# ==================================================

X = dataset.drop("personality", axis=1)
y = dataset["personality"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==================================================
# MODEL TRAINING
# ==================================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================================================
# CONFUSION MATRIX
# ==================================================

plt.figure(figsize=(7, 5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(IMAGE_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# ==================================================
# FEATURE IMPORTANCE
# ==================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature",
    palette="viridis"
)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=300, bbox_inches="tight")
plt.show()


# ==================================================
# TESTING NEW SAMPLES
# ==================================================

new_data = np.array([
    [0, 5, 0, 0, 1.2, 2, 0, 3],
    [3, 20, 4, 5, 12.5, 8, 4, 2],
    [1, 12, 2, 2, 6.0, 7, 1, 2]
])

pred = model.predict(new_data)

print("\nPredicted Personality Types:")
print(le.inverse_transform(pred))
