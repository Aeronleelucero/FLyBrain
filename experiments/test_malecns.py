"""FLY-CODER: Test the MaleCNS connectome backend."""

from flybrain import FlyBrain


def main():
    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    print("Dataset provenance:")
    print(fly.provenance)

    print("\nInitial activity:")
    print(fly.activity())

    print("\nApplying visual stimulus...")
    fly.sense.vision(left=1.0, right=0.0)

    fly.run(ms=100)

    print("\nActivity after 100 ms:")
    print(fly.activity())

    print("\nDescending output:")
    print(fly.output("descending"))

    print("\nMotor proxy:")
    print(fly.motor())


if __name__ == "__main__":
    main()
