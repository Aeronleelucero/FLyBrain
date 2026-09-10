"""Usage: python examples/image/main.py photo.png --dataset malecns"""
import argparse
import numpy as np
from PIL import Image
from flybrain import FlyBrain

parser = argparse.ArgumentParser()
parser.add_argument("image")
parser.add_argument("--dataset", default="synthetic")
args = parser.parse_args()
brain = FlyBrain(args.dataset)
with Image.open(args.image) as image:
    brain.vision(np.asarray(image.convert("RGB")))
brain.run(100)
print(brain.output("descending"))
