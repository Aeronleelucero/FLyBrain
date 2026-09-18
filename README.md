# FlyBrain 🪰

**Plug a fruit-fly connectome into anything.**

FlyBrain is a Python SDK for connecting public fruit-fly wiring to games, media,
robots and data streams. It is **not a digitized conscious fruit fly**.

| Real source data | Modeled / assumed |
| --- | --- |
| Reconstructed connectivity and synapse counts | LIF membrane and temporal dynamics |
| Available cell annotations and transmitter predictions | Synaptic gain and transmitter-to-sign transformation |
| Source IDs and provenance | Sensory encoding and motor decoding; no plasticity |

This repository is an independent project. Distribution name: `flybrain-sdk`,
Python import and CLI: `flybrain`. It has not been published to PyPI; install this
checkout instead of assuming that `pip install flybrain` installs this project.

```bash
pip install -e .
flybrain pull malecns
```

```python
from flybrain import FlyBrain

fly = FlyBrain("malecns")
fly.sense.vision(left=1.0)
fly.run(100)
print(fly.output("descending"))
print(fly.motor())  # Explicit, uncalibrated population-rate proxies
```

Use `fly.vision(frame)` for a NumPy grayscale/RGB frame (uint8 0–255 or float 0–1).
The MVP reduces each frame to left/right mean illumination; it does not implement
a retinotopic compound eye. For an immediate offline demo:

```bash
flybrain run examples/hello_brain.py
```

See [API guide](docs/API.md), [data and licenses](docs/DATASETS.md),
[research](docs/RESEARCH.md), [architecture](ARCHITECTURE.md), and
[agent integration rules](AGENTS.md).

## License & Usage

This repository is intentionally public so people can inspect, clone, run, and
privately modify FlyBrain for personal, educational, and non-commercial research
use. Public visibility does not mean unrestricted commercial rights.

FlyBrain is publicly available for personal, educational, and non-commercial
research use. You may clone and modify your own copy under the terms of the
FlyBrain license. Commercial use, redistribution, repackaging, and presenting
FlyBrain as your own project require explicit permission from the copyright
holder.

Private forks and local copies are allowed only within the license terms. The
official FlyBrain project and repository are maintained by the original author.
The FlyBrain name, branding, and official-project identity may not be used to
imply that a fork or modification is official. Read the custom
[source-available license](LICENSE), which is not an OSI-approved open-source
license, before using the project.

## Included in this MVP

- Official MaleCNS and FlyWire loaders, verified downloads, sparse caches and provenance.
- Deterministic CPU LIF, semantic vision/smell/touch, inspectable mappings and output proxies.
- Adapter protocol, image/video/webcam and closed-loop world examples.
- CLI, local REST/OpenAPI server, CSV logging and optional activity plots.
- Offline tests, CI and reproducible benchmark scripts.

MaleCNS end-to-end verification retained **166,700 neurons and 25,582,938 edges**.
On the development Windows environment, 100 simulated ms took approximately
5.1 seconds; this is not a real-time performance claim. See
[validation report](docs/VALIDATION.md) for scope, measurements and remaining limits.

```bash
pip install -e '.[dev]'
python -m pytest -q
python benchmarks/benchmark.py --dataset synthetic --ms 1000
```

Optional extras: `.[server]`, `.[media]`, `.[plot]`. Use `python -m flybrain`
if the console script directory is absent from PATH. CUDA, calibrated sensory
and motor models, plasticity, morphology/3D viewing and PyPI publication are
future work. This project does not claim compatibility with the unrelated
existing browser project also named FlyBrain.
