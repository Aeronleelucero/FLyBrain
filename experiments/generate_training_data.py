"""Generate MaleCNS activity data for classifier training."""

import csv
from pathlib import Path

from flybrain import FlyBrain


OUTPUT_FILE = Path("data/malecns_training_data.csv")


def get_action_label(left, right):
    """
    Create an initial training label from the vision input.

    This is a teacher rule for generating data.
    Later, you can replace this with labels from a
    Pygame environment or human demonstrations.
    """

    if left == 0.0 and right == 0.0:
        return "idle"

    difference = left - right

    if left < 0.5 and right < 0.5:
        return "weak_move"

    if difference >= 0.20:
        return "move_left"

    if difference <= -0.20:
        return "move_right"

    return "move_forward"


def simulate_case(left, right):
    """Run MaleCNS and return its output features."""

    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    fly.sense.vision(
        left=left,
        right=right,
    )

    fly.run(ms=35)

    descending = fly.output("descending")

    mean_rate = descending["mean_rate_hz"] or 0.0
    left_rate = descending["left_rate_hz"] or 0.0
    right_rate = descending["right_rate_hz"] or 0.0
    difference_hz = left_rate - right_rate

    return {
        "vision_left": left,
        "vision_right": right,
        "mean_rate_hz": mean_rate,
        "left_rate_hz": left_rate,
        "right_rate_hz": right_rate,
        "difference_hz": difference_hz,
        "action": get_action_label(left, right),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Generate many combinations automatically.
    values = [
    0.0,
    0.01,
    0.10,
    0.25,
    0.40,
    0.50,
    0.505,
    0.52,
    0.60,
    0.75,
    1.0,
]
    

    total = len(values) * len(values)

    print(f"Generating {total} MaleCNS samples...")

    for index, left in enumerate(values):
        for right in values:
            row = simulate_case(left, right)
            rows.append(row)

        print(
            f"Progress: {index + 1}/{len(values)} "
            f"left-input levels completed"
        )

    fieldnames = [
        "vision_left",
        "vision_right",
        "mean_rate_hz",
        "left_rate_hz",
        "right_rate_hz",
        "difference_hz",
        "action",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved {len(rows)} samples to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
