"""FLY-CODER: Compare visual inputs using the real MaleCNS dataset."""

from flybrain import FlyBrain


def run_case(name, left, right):
    print("\n" + "=" * 70)
    print(name)
    print(f"Input: left={left}, right={right}")

    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    fly.sense.vision(left=left, right=right)
    fly.run(ms=100)

    descending = fly.output("descending")
    motor = fly.motor()

    print("\nDescending:")
    print(descending)

    print("\nMotor:")
    print(motor)

    return {
        "name": name,
        "descending": descending,
        "motor": motor,
    }


def main():
    cases = [
        ("Left vision", 1.0, 0.0),
        ("Right vision", 0.0, 1.0),
        ("Both sides", 1.0, 1.0),
        ("No vision", 0.0, 0.0),
        ("Balanced weak vision", 0.5, 0.5),
        ("Strong left / weak right", 1.0, 0.25),
        ("Weak left / strong right", 0.25, 1.0),
    ]

    results = []

    for name, left, right in cases:
        results.append(run_case(name, left, right))

    print("\n" + "=" * 70)
    print(f"Experiment complete. Cases tested: {len(results)}")


if __name__ == "__main__":
    main()
