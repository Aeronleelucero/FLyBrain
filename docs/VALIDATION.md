# Validation report

Local run: 2026-09-11 Asia/Seoul, Windows 11, Python 3.12.14,
NumPy 2.3.5, SciPy 1.18.1, PyArrow 25.0.1. Timings are one local observation,
not a throughput guarantee or cross-machine benchmark.

## Offline

`python -m pytest -q`: **24 passed**. Coverage includes source-schema parsing,
duplicate edge aggregation, exclusion accounting, cache integrity, failed download
checksums, graph shape and IDs, analytic leak, reset/refractory, inhibition,
chunked determinism, semantic image and modality mapping, adapter boundaries,
CLI, REST validation and pagination. Two dependency deprecation warnings were
reported by the installed Starlette/HTTPX test-client integration; no tests failed.

`python tools/server_check.py`: passed a real CLI-launched server and TCP JSON
stimulus → step → output flow. Server terminated after the check.

`python -m compileall -q src examples tools benchmarks`: passed.

Editable installation and `python -m pip wheel . --no-deps -w dist` succeeded.
The headless 2D world example completed 100 encode/simulate/decode cycles.

Synthetic benchmark, 10 neurons / 8 edges, 1000 simulated ms: about 0.031 seconds
of simulation, 140 bytes of topology sparse buffers. Do not extrapolate this
tiny fixture to whole-connectome runtime.

## MaleCNS real-data integration

`python tools/real_check.py malecns` downloaded official v1.0 files and completed
preprocessing and the requested API sequence on the complete retained population:

| Measurement | Observed |
| --- | ---: |
| Source annotation rows | 211,577 |
| Retained neurons (nonempty superclass) | 166,700 |
| Source connection rows | 151,856,684 |
| Retained directed edges | 25,582,938 |
| Excluded edge rows (outside retained population) | 126,273,746 |
| Excluded synapses | 187,655,626 |
| Raw-count CSR buffer bytes | 307,662,060 |
| Cache load including verification and engine construction | 7.83 s |
| 100 simulated ms, CPU, dt=1 ms | 5.14 s |
| Mapped left / right photoreceptors | 2,345 / 3,746 |
| Total modeled spikes after left illumination | 453,068 |
| Mean descending activity | 39.57 Hz |

Input: `sense.vision(left=1.0)`, seed=0, default parameters. Rates accumulate since
reset. These numbers demonstrate a functioning pipeline, not validated biological
behavior. The topology byte count excludes annotation objects, model weights,
temporary conversion buffers and Python/runtime memory. Preprocessing scans all
151.9 million source rows and can take minutes.

The retained graph includes isolated classified neurons. The original table
contains unclassified segments; the manifest explicitly accounts for filtering.
Original files, checksums and processed caches remain outside the repository.

## Scope limits

FlyWire: the pinned 139,255-row annotation file was downloaded and its actual
column names/transmitter fields checked. The official Zenodo file API returned
valid Arrow bytes for a range request, but 4 MB took approximately 72 seconds in
this environment. The full 852 MB transfer was stopped without publishing a
processed cache. **FlyWire whole-graph end-to-end execution remains unverified**.
Its loader uses the official pinned MD5 and is covered by source-schema fixtures;
run `python tools/real_check.py flywire` to complete that integration check on a
usable connection. There is no synthetic fallback or claim that a FlyWire cache
was prepared.

CUDA is not implemented. No webcam was opened. Media examples require optional
dependencies and user media. No Unity/Godot/game engine plugin was tested.
No learning, biological behavior, motor calibration, statistical neuroscience
validation or real-time full-connectome performance is asserted.
