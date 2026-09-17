"""FLY-CODER: Compare visual inputs."""

from flybrain import FlyBrain


def run_case(name, left, right):
    fly = FlyBrain("synthetic", seed=0)

    fly.sense.vision(left=left, right=right)
    fly.run(ms=100)

    activity = fly.activity()
    descending = fly.output("descending")
    motor = fly.motor()

    print(f"\n{'=' * 60}")
    print(name)
    print(f"Input: left={left}, right={right}")
    print(f"Time: {activity['time_ms']} ms")
    print(f"Spike counts: {activity['spike_counts']}")
    print(f"Rates: {activity['rates_hz']}")
    print(f"Descending: {descending}")
    print(f"Motor: {motor}")

    return {
        "name": name,
        "spike_counts": activity["spike_counts"],
        "rates_hz": activity["rates_hz"],
        "descending": descending,
        "motor": motor,
    }


def main():
    cases = [
        ("Left vision", 1.0, 0.0),
        ("Right vision", 0.0, 1.0),
        ("Both sides", 1.0, 1.0),
        ("No vision", 0.0, 0.0),
    ]

    results = []

    for name, left, right in cases:
        results.append(run_case(name, left, right))

    print("\nExperiment complete.")
    print(f"Cases tested: {len(results)}")


if __name__ == "__main__":
    main()
