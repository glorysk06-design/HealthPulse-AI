import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


def main():
    # Find the project files relative to this script.
    project_folder = Path(__file__).resolve().parent
    data_path = project_folder / "data" / "diabetes.csv"
    model_folder = project_folder / "models"
    model_path = model_folder / "health_model.pkl"
    importance_path = model_folder / "feature_importances.json"

    # The CSV has no header, so assign the column names while loading it.
    column_names = FEATURE_NAMES + ["Outcome"]
    data = pd.read_csv(data_path, header=None, names=column_names)

    # Separate the input features from the target column.
    features = data.drop(columns="Outcome")
    target = data["Outcome"]

    # Keep a portion of the data aside to evaluate the trained model.
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Fit a temporary model for a fair holdout evaluation. Do not replace the
    # saved model used by the Flask application.
    evaluation_model = RandomForestClassifier(random_state=42)
    evaluation_model.fit(x_train, y_train)

    # Print evaluation results for the test data.
    predictions = evaluation_model.predict(x_test)
    print("Evaluation Metrics (test set)")
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # Save importance values from the existing deployed model so they describe
    # the same model that the Flask application uses. Importance is not causation.
    model_folder.mkdir(parents=True, exist_ok=True)
    deployed_model = joblib.load(model_path)
    importance_rows = sorted(
        zip(FEATURE_NAMES, deployed_model.feature_importances_),
        key=lambda row: row[1],
        reverse=True,
    )
    importance_data = {
        "note": (
            "Relative Random Forest feature importance values; these do not "
            "show that a feature causes a prediction."
        ),
        "features": [
            {"name": name, "importance": float(value)}
            for name, value in importance_rows
        ],
    }
    with importance_path.open("w", encoding="utf-8") as importance_file:
        json.dump(importance_data, importance_file, indent=2)

    print("\nFeature Importance (existing saved model, descending)")
    print("These relative values do not indicate causation.")
    for name, value in importance_rows:
        print(f"{name:<30} {value:.6f} ({value * 100:.2f}%)")
    print(f"\nFeature importance saved to: {importance_path}")
    print(f"Existing model left unchanged: {model_path}")


if __name__ == "__main__":
    main()