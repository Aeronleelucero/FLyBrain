"""Run a continuous MaleCNS agent using the trained action classifier."""

import time
from pathlib import Path

import joblib
import pandas as pd

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


def load_model():
    """Load the trained Random Forest model."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run the training script first."
        )

    print(f"Loading model: {MODEL_FILE}")

    return joblib.load(MODEL_FILE)


def extract_features(left, right, descending):
    """Build the feature vector used during training."""

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


def predict_action(model, left, right):
    """Run MaleCNS and predict an action."""

    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    # Send simulated vision input into MaleCNS.
    fly.sense.vision(
        left=left,
        right=right,
    )

    # Run the neural simulation.
    fly.run(ms=50)

    # Read descending-neuron output.
    descending = fly.output("descending")

    # Extract the same features used during training.
    features = extract_features(
        left=left,
        right=right,
        descending=descending,
    )

    # Use a DataFrame with the original feature names.
    feature_frame = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    # Safety rule:
    # If MaleCNS activity is zero or very low,
    # the agent must remain idle.
    if features[2] < 5.0:
        action = "idle"
    else:
        action = model.predict(feature_frame)[0]

    return {
        "action": action,
        "mean_rate_hz": features[2],
        "left_rate_hz": features[3],
        "right_rate_hz": features[4],
        "difference_hz": features[5],
    }


def generate_vision_input(step):
    """
    Generate a repeating simulated vision pattern.

    This is only a test environment.
    It does not use a real camera yet.
    """

    phase = step % 40

    # No detected obstacle.
    if phase < 8:
        return 0.0, 0.0, "no vision"

    # Obstacle on the left.
    if phase < 16:
        return 0.60, 0.0, "left obstacle"

    # Obstacle on the right.
    if phase < 24:
        return 0.01, 0.80, "right obstacle"

    # Obstacle directly ahead.
    if phase < 32:
        return 1.0, 1.0, "front obstacle"

    # Clear environment.
    return 0.0, 0.0, "clear"


def main():
    """Run the continuous MaleCNS agent."""

    model = load_model()

    print()
    print("Starting live trained MaleCNS agent.")
    print("Press Ctrl+C to stop.")
    print()

    step = 0

    try:
        while True:
            # Generate simulated sensory input.
            left, right, situation = generate_vision_input(step)

            # Run the brain and classify the result.
            result = predict_action(
                model=model,
                left=left,
                right=right,
            )

            # Display the current agent state.
            print(
                f"step={step:04d} | "
                f"situation={situation:15} | "
                f"left_input={left:4.2f} | "
                f"right_input={right:4.2f} | "
                f"mean={result['mean_rate_hz']:7.3f} Hz | "
                f"diff={result['difference_hz']:7.3f} Hz | "
                f"action={result['action']}"
            )

            step += 1

            # Small delay to make the output readable.
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nAgent stopped safely.")


if __name__ == "__main__":
    main()