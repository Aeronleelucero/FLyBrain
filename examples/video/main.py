"""Usage: python examples/video/main.py video.mp4 --dataset malecns --frames 100"""
import argparse
import cv2
from flybrain import FlyBrain

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--dataset", default="synthetic")
parser.add_argument("--frames", type=int, default=100)
args = parser.parse_args()
brain = FlyBrain(args.dataset)
capture = cv2.VideoCapture(args.video)
try:
    if not capture.isOpened(): raise RuntimeError("Could not open video")
    for index in range(args.frames):
        ok, frame = capture.read()
        if not ok: break
        brain.vision(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        brain.step(10)  # Deliberately fixed simulated time, independent of source FPS
        print(index, brain.output("descending"))
finally:
    capture.release()
