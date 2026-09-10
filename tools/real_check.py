"""Explicit integration check; downloads public data outside the repository."""
import json
import sys
import time
from flybrain import FlyBrain, pull

dataset = sys.argv[1] if len(sys.argv) > 1 else "malecns"
pull(dataset, progress=lambda message: print(message, flush=True))
start = time.perf_counter()
fly = FlyBrain(dataset)
loaded = time.perf_counter()
fly.sense.vision(left=1.0)
fly.run(100)
result = {"dataset": dataset, "neurons": len(fly.connectome.neurons),
          "edges": fly.connectome.counts.nnz, "sparse_bytes": fly.connectome.sparse_bytes,
          "load_seconds": loaded-start, "run_100ms_seconds": time.perf_counter()-loaded,
          "spikes": int(fly.engine.counts.sum()), "descending": fly.output("descending"),
          "vision_left_count": len(fly._mapping("vision.left")),
          "vision_right_count": len(fly._mapping("vision.right")),
          "preprocessing": fly.connectome.provenance["preprocessing"]}
print(json.dumps(result, indent=2), flush=True)
