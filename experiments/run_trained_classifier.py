"""Run MaleCNS and automatically classify its output using a trained model."""

from pathlib import Path

import joblib

from flybrain import FlyBrain


MODEL_FILE = Path("models/malecns_action_classifier.joblib")


FEATURE_NAMES = [
    "vision_left",
    "vision_right",
    "mean_rate_hz",
    "left_rate_hz",
    "right_rate_hz",
    "difference_hz",
]


def load_trained_model():
    """Load the saved machine-learning model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run train_action_model.py first."
        )

    print(f"Loading trained model: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


def extract_features(left, right, descending):
    """Convert MaleCNS output into model features."""

    mean_rate = descending["mean_rate_hz"] or 0.0
    left_rate = descending["left_rate_hz"] or 0.0
    right_rate = descending["right_rate_hz"] or 0.0
    difference_hz = left_rate - right_rate

    return [
        left,
        right,
        mean_rate,
        left_rate,
        right_rate,
        difference_hz,
    ]


def run_case(name, left, right, model):
    """Run one MaleCNS test and predict the action."""

    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    # Send vision input to MaleCNS.
    fly.sense.vision(
        left=left,
        right=right,
    )

    # Run the neural simulation.
    fly.run(ms=50)

    # Get descending-neuron activity.
    descending = fly.output("descending")

    # Convert the output into classifier features.
    features = extract_features(
        left=left,
        right=right,
        descending=descending,
    )

    # Predict using the trained model.
    prediction = model.predict([features])[0]

    # Get prediction probabilities when supported.
    probabilities = {}

    if hasattr(model, "predict_proba"):
        probability_values = model.predict_proba([features])[0]

        probabilities = dict(
            zip(
                model.classes_,
                probability_values,
            )
        )

    mean_rate = features[2]
    left_rate = features[3]
    right_rate = features[4]
    difference_hz = features[5]

    print(
        f"{name:25} | "
        f"mean={mean_rate:7.3f} Hz | "
        f"left={left_rate:7.3f} Hz | "
        f"right={right_rate:7.3f} Hz | "
        f"diff={difference_hz:7.3f} Hz | "
        f"action={prediction}"
    )

    if probabilities:
        confidence = probabilities[prediction] * 100

        print(
            f"{'':25} | "
            f"confidence={confidence:6.2f}%"
        )


def main():
    """Load the model and run automatic predictions."""

    model = load_trained_model()

    print()
    print(
        f"{'Case':25} | "
        f"{'Mean':>12} | "
        f"{'Left':>12} | "
        f"{'Right':>12} | "
        f"{'Diff':>12} | Action"
    )

    print("-" * 105)

    cases = [
        ("No vision", 0.0, 0.0),
        ("Weak left", 0.52, 0.0),
        ("Medium left", 0.60, 0.0),
        ("Strong left", 1.0, 0.0),
        ("Weak right", 0.0, 0.52),
        ("Medium right", 0.0, 0.60),
        ("Strong right", 0.0, 1.0),
        ("Both sides", 1.0, 1.0),
        ("Right dominant", 0.20, 1.0),
        ("Right only high", 0.01, 1.0),
        ("Right only medium", 0.01, 0.80),
    ]

    for name, left, right in cases:
        run_case(
            name=name,
            left=left,
            right=right,
            model=model,
        )


if __name__ == "__main__":
    main()
