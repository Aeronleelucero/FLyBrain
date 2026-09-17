"""Convert MaleCNS descending activity into simple application actions."""

from flybrain import FlyBrain


# Application-level thresholds.
IDLE_THRESHOLD_HZ = 5.0
MOVE_THRESHOLD_HZ = 20.0
TURN_THRESHOLD_HZ = 2.0


def classify_output(descending, motor=None):
    """
    Convert descending-neuron activity into a simple action.

    Actions:
        idle         - No meaningful activity.
        weak_move    - Weak activity.
        move_forward - Strong activity without clear turning bias.
        move_left    - Left activity is significantly stronger.
        move_right   - Right activity is significantly stronger.
    """

    mean_rate = descending["mean_rate_hz"] or 0.0
    left_rate = descending["left_rate_hz"] or 0.0
    right_rate = descending["right_rate_hz"] or 0.0

    # Calculate the left-right activity difference.
    difference_hz = left_rate - right_rate

    # No meaningful activity.
    if mean_rate < IDLE_THRESHOLD_HZ:
        return "idle"

    # Activity exists, but it is not strong enough for normal movement.
    if mean_rate < MOVE_THRESHOLD_HZ:
        return "weak_move"

    # Clear left-side dominance.
    if difference_hz >= TURN_THRESHOLD_HZ:
        return "move_left"

    # Clear right-side dominance.
    if difference_hz <= -TURN_THRESHOLD_HZ:
        return "move_right"

    # Strong activity without a clear turning direction.
    return "move_forward"


def run_case(name, left, right):
    """Run one vision input case and display the result."""

    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    # Send vision input into the MaleCNS model.
    fly.sense.vision(
        left=left,
        right=right,
    )

    # Run the simulation.
    fly.run(ms=100)

    # Read model outputs.
    descending = fly.output("descending")
    motor = fly.motor()

    # Calculate the left-right descending activity difference.
    left_rate = descending["left_rate_hz"] or 0.0
    right_rate = descending["right_rate_hz"] or 0.0
    difference_hz = left_rate - right_rate

    # Convert the model output into an application action.
    action = classify_output(
        descending=descending,
        motor=motor,
    )

    # Display the result.
    print(
        f"{name:25} | "
        f"mean={descending['mean_rate_hz']:7.3f} Hz | "
        f"left={left_rate:7.3f} Hz | "
        f"right={right_rate:7.3f} Hz | "
        f"diff={difference_hz:7.3f} Hz | "
        f"action={action}"
    )


def main():
    """Run all test cases."""

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
        )


if __name__ == "__main__":
    main()