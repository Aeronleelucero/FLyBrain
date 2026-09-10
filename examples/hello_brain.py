"""Offline smoke demo. Change synthetic to malecns after flybrain pull malecns."""
from flybrain import FlyBrain

fly = FlyBrain("synthetic", seed=0)
fly.sense.vision(left=1.0)
fly.run(100)
print(fly.activity())
print(fly.output("descending"))
print(fly.motor())
