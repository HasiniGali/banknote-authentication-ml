# ============================================================
# BANKNOTE AUTHENTICATION USING MACHINE LEARNING
# Models: Decision Tree and K-Nearest Neighbors (KNN)
# ============================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ------------------------------------------------------------
# 1. CREATE PROJECT FOLDERS
# ------------------------------------------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("=" * 60)
print("BANKNOTE AUTHENTICATION USING MACHINE LEARNING")
print("=" * 60)


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

print("\n[1/5] Loading the dataset...")

try:
    banknote = fetch_ucirepo(id=267)

    X = banknote.data.features
    y = banknote.data.targets.iloc[:, 0]

    # Combine features and target
    data = pd.concat(
        [X, y.rename("class")],
        axis=1
    )

    # Remove missing values if any exist
    data = data.dropna()

    # Separate features and target
    X = data.drop(columns=["class"])
    y = data["class"]

    print("Dataset loaded successfully!")
    print("Total records:", len(data))
    print("Features:", list(X.columns))
    print("Missing values:", int(data.isnull().sum().sum()))

    # Save dataset locally
    data.to_csv(
        "results/banknote_dataset.csv",
        index=False
    )

except Exception as error:
    print("\nCould not load the dataset.")
    print("Check your internet connection and try again.")
    print("Error:", error)
    raise SystemExit(1)


# ------------------------------------------------------------
# 3. SPLIT THE DATASET
# ------------------------------------------------------------

print("\n[2/5] Splitting the dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# ------------------------------------------------------------
# 4. BUILD THE MACHINE LEARNING MODELS
# ------------------------------------------------------------

print("\n[3/5] Preparing machine learning models...")

models = {
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ])
}


# ------------------------------------------------------------
# 5. TRAIN AND EVALUATE THE MODELS
# ------------------------------------------------------------

print("\n[4/5] Training and evaluating models...")

results = []
class_labels = sorted(y.unique())

for name, model in models.items():

    print("\n" + "-" * 50)
    print("Model:", name)
    print("-" * 50)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    predictions = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Display classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # Save trained model
    filename = name.lower().replace(" ", "_")

    model_path = f"models/{filename}.joblib"

    joblib.dump(model, model_path)

    print("Saved model:", model_path)

    # Generate confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_labels
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    plt.savefig(
        f"results/{filename}_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # Store evaluation results
    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy * 100, 2)
    })


# ------------------------------------------------------------
# 6. COMPARE MODEL PERFORMANCE
# ------------------------------------------------------------

print("\n[5/5] Comparing model performance...")

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print(results_df.to_string(index=False))


# Generate model comparison graph
plt.figure(figsize=(8, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy (%)"
)

plt.title("Banknote Authentication - Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "results/model_comparison.png",
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 7. SAVE FINAL SUMMARY
# ------------------------------------------------------------

best_model_row = results_df.loc[
    results_df["Accuracy (%)"].idxmax()
]

summary = f"""
BANKNOTE AUTHENTICATION PROJECT
================================

Dataset: UCI Banknote Authentication
Total Records: {len(data)}
Number of Features: {X.shape[1]}
Training Records: {len(X_train)}
Testing Records: {len(X_test)}

MODEL RESULTS
-------------

{results_df.to_string(index=False)}

Best Model by Test Accuracy:
{best_model_row['Model']}

Best Test Accuracy:
{best_model_row['Accuracy (%)']}%

Note:
This model demonstrates banknote classification
using the UCI dataset. It is not specifically
trained to authenticate Indian currency.
"""

with open("results/summary.txt", "w", encoding="utf-8") as file:
    file.write(summary)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print("\nGenerated files:")
print("1. models/decision_tree.joblib")
print("2. models/knn.joblib")
print("3. results/banknote_dataset.csv")
print("4. results/model_comparison.csv")
print("5. results/model_comparison.png")
print("6. results/decision_tree_confusion_matrix.png")
print("7. results/knn_confusion_matrix.png")
print("8. results/summary.txt")

print("\nYour models are trained and saved.")
print("You can now prepare the project for GitHub.")