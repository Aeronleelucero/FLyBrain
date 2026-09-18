# Dataset provenance and licenses

FlyBrain code is available under its custom source-available license; downloaded data retains its own terms and attribution. No real
connectome is distributed in this repository or package.

| Backend | Version | Source/terms |
| --- | --- | --- |
| `malecns` | 1.0, min confidence 0.5 | [Janelia bulk data](https://male-cns.janelia.org/download/), CC-BY-4.0 |
| `flywire` / `fafb` | FAFB 783; annotation release 2.1.0 | [Zenodo connectivity](https://zenodo.org/records/10676866), CC-BY-4.0 for this deposit; [public annotation guidelines](https://flywire.ai/guidelines), CC BY-NC 4.0 |
| `synthetic` | 1 | Handwritten 10-neuron fixture, covered by the FlyBrain license; no biological provenance claim |

MaleCNS includes the CNS; FlyWire FAFB is the female brain and is not a complete
motor/body connectome. Descending activity is a readout of annotated populations.

## Download and inspect

```bash
flybrain datasets
flybrain pull malecns
flybrain info malecns
flybrain pull flywire
```

Raw data is approximately 1.1 GB for MaleCNS and 0.9 GB for FlyWire; reserve several
GB of disk and RAM for decompression, sparse conversion and caches. Final footprint
and timing depend on the environment. Files stay under `~/.cache/flybrain` by
default. Override with `FLYBRAIN_CACHE` or `flybrain --cache-dir PATH pull malecns`.
The constructor reads a verified cache; it never silently starts a large download.

The downloader streams into `.part` files and verifies Content-Length plus GCS
MD5 or pinned Zenodo MD5 when available. SHA256, exact URL, retrieval time, size,
ETag and verification method are recorded per file. When a source has no upstream
checksum, the manifest says so: a locally computed SHA256 is not independent
authentication. GitHub annotation URLs are pinned to a commit. A failed download
is retried on the next pull, restarting the incomplete file; byte-range resume
is not implemented.

Processed caches contain `counts.npz`, `neurons.json` and `manifest.json`.
The manifest is published last, after preprocessing, and includes SHA256 of both
processed files. Every load checks these hashes. Concurrent pulls for the same
dataset are rejected through a lock file. After an interrupted process, remove
only its stale `<dataset>.lock` after confirming the process has exited.

## What preprocessing changes

- MaleCNS retains rows with a nonempty `superclass` annotation, excluding unlabeled
  segments. Original annotation rows and excluded counts are reported.
- FlyWire retains all rows in the pinned neuron annotation table.
- Only edges with both endpoints in the retained population enter the graph.
  Excluded edge rows and synapse counts are recorded, including zero-degree neurons.
- Duplicate neuron pairs (including different neuropil rows) are summed. Self edges
  remain. No synapse threshold is added beyond the source release's filtering.
- Matrix convention: `counts[postsynaptic, presynaptic]`. Counts are positive
  integers, independent of all dynamics. Arrow batches and a temporary binary spool
  avoid a Python object per edge; the final sparse conversion is still O(E) RAM.
- Laterality: use explicit `side`, then `rootSide`, then `soma_side`/`somaSide`.
  Sensory roots frequently lack a soma location. The source fields remain exposed.
- MaleCNS transmitter policy uses `consensus_nt`, falling back to `predicted_nt`.
  Source confidence and ground-truth fields remain in `transmitter_annotations`.
  The LIF sign policy is a separate, inspectable assumption.
- Morphology is not downloaded by the MVP. The source URLs and IDs permit future
  Neuroglancer/navis integrations without claiming a current 3D renderer.

No automatic schema guessing or silent synthetic replacement occurs on failure.
If upstream columns change, import errors report the missing columns.

## Citation

Use the publications and dataset versions relevant to the selected backend:

- MaleCNS: [Berg et al., project and publication links](https://male-cns.janelia.org/),
  [preprint DOI](https://doi.org/10.1101/2025.10.09.680999), dataset v1.0.
- FlyWire topology: [Dorkenwald et al., Nature 2024](https://doi.org/10.1038/s41586-024-07558-y).
- FlyWire annotations: [Schlegel et al., Nature 2024](https://doi.org/10.1038/s41586-024-07686-5).
- Connectivity deposit: [FlyWire Consortium, Zenodo 10676866](https://doi.org/10.5281/zenodo.10676866).
- Transmitter predictions: [Eckstein et al., Cell 2024](https://doi.org/10.1016/j.cell.2024.03.016).
- LIF scientific background, not this implementation's validation:
  [Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9).
