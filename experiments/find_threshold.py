"""Find the approximate MaleCNS visual activation threshold."""

from flybrain import FlyBrain


def test_intensity(intensity):
    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    fly.sense.vision(left=intensity, right=0.0)
    fly.run(ms=100)

    descending = fly.output("descending")
    motor = fly.motor()

    return descending, motor


def main():
    print("Testing activation threshold...\n")

    for intensity_index in range(500, 601, 5):
        intensity = intensity_index / 1000

        descending, motor = test_intensity(intensity)

        print(
            f"intensity={intensity:.3f} | "
            f"mean={descending['mean_rate_hz']:.3f} Hz | "
            f"forward={motor.forward:.3f} | "
            f"turn_left={motor.turn_left:.3f}"
        )


if __name__ == "__main__":
    main()
