"""Experimental arbitrary numeric stream encoding. Not an investment model.

No market API, orders, prices or financial predictions. Replace the generated
wave with any normalized scalar stream to observe modeled activity.
"""
import math
from flybrain import FlyBrain

brain = FlyBrain("synthetic")
for tick in range(30):
    value = (1+math.sin(tick/5))/2
    brain.sense.vision(left=value, right=1-value)
    brain.step(10)
    print(tick, value, brain.output("descending"))
