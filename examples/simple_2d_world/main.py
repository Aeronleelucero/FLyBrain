"""Headless closed-loop point world; CSV output, no biological locomotion claim."""
import math
from flybrain import FlyBrain, interact


class WorldAdapter:
    def encode(self, environment):
        bearing = math.atan2(environment["light_y"]-environment["y"], environment["light_x"]-environment["x"])
        difference = math.atan2(math.sin(bearing-environment["heading"]), math.cos(bearing-environment["heading"]))
        return {"vision": {"left": max(0.0, math.cos(difference-.5)),
                           "right": max(0.0, math.cos(difference+.5))}}

    def decode(self, brain_output):
        left = brain_output["left_rate_hz"] or 0
        right = brain_output["right_rate_hz"] or 0
        return {"speed": min(brain_output["mean_rate_hz"]/100, 1),
                "rotation": (left-right)/100}


if __name__ == "__main__":
    brain = FlyBrain("synthetic")
    world = dict(x=0.,y=0.,heading=0.,light_x=10.,light_y=5.)
    print("tick,x,y,heading")
    for tick in range(100):
        action = interact(brain, WorldAdapter(), world, ms=10)
        world["heading"] += action["rotation"]*.1
        world["x"] += action["speed"]*math.cos(world["heading"])*.1
        world["y"] += action["speed"]*math.sin(world["heading"])*.1
        print(f"{tick},{world['x']:.4f},{world['y']:.4f},{world['heading']:.4f}")
