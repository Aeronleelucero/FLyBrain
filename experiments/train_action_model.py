"""Train a machine-learning action classifier from MaleCNS outputs."""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# Project paths
DATA_FILE = Path("data/malecns_training_data.csv")
MODEL_FILE = Path("models/malecns_action_classifier.joblib")


# Input features used by the classifier
FEATURES = [
    "vision_left",
    "vision_right",
    "mean_rate_hz",
    "left_rate_hz",
    "right_rate_hz",
    "difference_hz",
]


# Target/output column
TARGET = "action"


def main():
    """Load data, train the classifier, evaluate it, and save it."""

    # Check whether the training dataset exists.
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Training data not found: {DATA_FILE}\n"
            "Run generate_training_data.py first."
        )

    # Load the CSV dataset.
    data = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(data)} training samples.")

    # Validate required columns.
    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The training dataset is missing these columns: "
            f"{missing_columns}"
        )

    # Remove rows with missing values.
    data = data.dropna(subset=required_columns)

    if data.empty:
        raise ValueError("The training dataset contains no usable rows.")

    # Separate input features and target labels.
    X = data[FEATURES]
    y = data[TARGET]

    print(f"Usable training samples: {len(data)}")

    print("\nAvailable action labels:")
    print(y.value_counts())

    # Check whether every class has at least two samples.
    class_counts = y.value_counts()

    if class_counts.min() >= 2:
        stratify_value = y
        print("\nUsing stratified train/test split.")
    else:
        stratify_value = None

        print(
            "\nWarning: At least one action class has fewer than "
            "2 samples."
        )
        print(
            "Using a non-stratified train/test split instead."
        )

    # Split the dataset into training and testing data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    # Create the Random Forest classifier.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    # Train the model.
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # Evaluate the model.
    predictions = model.predict(X_test)

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # Create the models directory if it does not exist.
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save the trained model.
    joblib.dump(model, MODEL_FILE)

    print()
    print(f"Saved trained model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()