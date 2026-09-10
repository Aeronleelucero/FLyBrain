"""Explicitly run to access camera; no recording or transmission. Ctrl+C exits."""
import argparse
import cv2
from flybrain import FlyBrain

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--dataset", default="synthetic")
parser.add_argument("--frames", type=int, default=100)
args = parser.parse_args()
brain = FlyBrain(args.dataset)
capture = cv2.VideoCapture(args.device)
try:
    if not capture.isOpened(): raise RuntimeError("Camera unavailable")
    for _ in range(args.frames):
        ok, frame = capture.read()
        if not ok: break
        brain.vision(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        brain.step(10)
        print(brain.output("descending"))
finally:
    capture.release()
