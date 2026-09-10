"""Offline default; use --dataset malecns after downloading for real profiling."""
import argparse
import json
import platform
import time
from flybrain import FlyBrain

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="synthetic")
parser.add_argument("--ms", type=float, default=1000)
args = parser.parse_args()
start=time.perf_counter()
brain=FlyBrain(args.dataset,seed=0)
loaded=time.perf_counter()
brain.sense.vision(left=1)
brain.run(args.ms)
finished=time.perf_counter()
print(json.dumps({"dataset":args.dataset,"python":platform.python_version(),
    "platform":platform.platform(),"neurons":len(brain.connectome.neurons),
    "edges":brain.connectome.counts.nnz,"topology_sparse_bytes":brain.connectome.sparse_bytes,
    "simulated_ms":args.ms,"load_seconds":loaded-start,"run_seconds":finished-loaded,
    "seed":0,"parameters":brain.provenance["parameters"]},indent=2))
