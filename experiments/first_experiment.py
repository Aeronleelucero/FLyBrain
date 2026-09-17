"""FLY-CODER: First custom neural experiment."""

from flybrain import FlyBrain


def run_experiment(left: float, right: float, duration_ms: float = 100):
    """Run a fly-brain simulation with visual input."""

    fly = FlyBrain("synthetic", seed=0)

    # Provide visual input.
    fly.sense.vision(left=left, right=right)

    # Run the simulation.
    fly.run(duration_ms)

    # Read the neural and motor outputs.
    activity = fly.activity()
    descending = fly.output("descending")
    motor = fly.motor()

    return activity, descending, motor


def main():
    experiments = [
        ("Left vision", 1.0, 0.0),
        ("Right vision", 0.0, 1.0),
        ("Both sides", 1.0, 1.0),
        ("No vision", 0.0, 0.0),
    ]

    for name, left, right in experiments:
        print(f"\n{'=' * 50}")
        print(name)
        print(f"Input: left={left}, right={right}")

        activity, descending, motor = run_experiment(left, right)

        print("Activity:", activity)
        print("Descending:", descending)
        print("Motor:", motor)


if __name__ == "__main__":
    main()
