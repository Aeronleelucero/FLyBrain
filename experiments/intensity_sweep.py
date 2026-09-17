"""FLY-CODER: Test MaleCNS response to different visual intensities."""

from flybrain import FlyBrain


def run_case(intensity):
    fly = FlyBrain(
        "malecns",
        dynamics="lif",
        seed=42,
    )

    fly.sense.vision(
        left=intensity,
        right=0.0,
    )

    fly.run(ms=100)

    descending = fly.output("descending")
    motor = fly.motor()

    print(
        f"intensity={intensity:.2f} | "
        f"mean={descending['mean_rate_hz']:.3f} Hz | "
        f"left={descending['left_rate_hz']:.3f} Hz | "
        f"right={descending['right_rate_hz']:.3f} Hz | "
        f"forward={motor.forward:.3f} | "
        f"turn_left={motor.turn_left:.3f} | "
        f"turn_right={motor.turn_right:.3f}"
    )


def main():
    for intensity in [
        0.0,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
    ]:
        run_case(intensity)


if __name__ == "__main__":
    main()

