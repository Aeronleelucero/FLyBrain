# Ecosystem review — 2026-09-11

Research preceded the runtime scaffold. No third-party simulator code was copied.

## Official data access

- [MaleCNS download page](https://male-cns.janelia.org/download/): public GCS
  Feather tables, CC-BY-4.0. Bulk download avoids the account/token required by
  neuPrint. Pin v1.0 and confidence threshold 0.5 source filenames. Download only
  annotations, per-body transmitter predictions and connection weights.
- [MaleCNS project](https://male-cns.janelia.org/): v1.0 released June 8, 2026;
  links the September 2026 publication and
  [Berg et al. preprint](https://doi.org/10.1101/2025.10.09.680999).
- [FlyWire guidelines](https://flywire.ai/guidelines): public-release licensing
  states CC BY-NC 4.0. Do not infer commercial permission for all components.
- [FlyWire official Zenodo record](https://zenodo.org/records/10676866): v783.0,
  explicitly CC-BY-4.0 on this connectivity deposit. Use
  `proofread_connections_783.feather`, with the record's published MD5. This
  component-level distinction is retained alongside the broader usage guidelines.
- [Research-team annotation repository](https://github.com/flyconnectome/flywire_annotations):
  links the same Zenodo deposit. Use v2.1.0, the 2024 paper annotations, pinned to
  commit `ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0`. Later releases add new
  cross-dataset interpretations; do not silently substitute them.
- Codex download API returned HTTP 403 in this environment. The loader therefore
  uses the official archived connectivity and research-team annotation sources.

## Existing simulation and game projects

| Project | License checked | Reuse decision |
| --- | --- | --- |
| [Shiu et al. model](https://github.com/philshiu/Drosophila_brain_model/blob/main/LICENSE) | MIT | Scientific reference; no code copied. Independent minimal LIF, not a reproduction of their calibrated model. |
| [DOOMFLY](https://github.com/nftechie/doomfly) | MIT repository; separate notices/data terms | Reference for application boundary and explicit failed validation reports. No game code, plasticity or button mapping imported. |
| [snedea/flybrain](https://github.com/snedea/flybrain) | MIT | Existing name collision and browser LIF experiment noted. Independent Python project; package distribution is `flybrain-sdk`. No assets or code copied. |

[Shiu et al., Nature 2024](https://doi.org/10.1038/s41586-024-07763-9)
supports investigating connectome-constrained LIF models, but does not validate
this SDK's parameters, sensory encoders or motor proxies. Game demonstrations
are engineering integrations, not evidence of consciousness or learned behavior.

## Build versus reuse

Reuse NumPy vector operations, SciPy sparse kernels, Arrow IPC readers, optional
FastAPI/Uvicorn and media libraries. Implement the small SDK facade, canonical
schema, provenance, download/cache lifecycle, semantic map explanations, LIF
state transition and application adapter boundary here. Avoid a mandatory
neuPrint/CAVE token, a 3D renderer, a body simulator or a reinforcement-learning
framework in the initial installation.

## Scientific limits carried into the design

Topology constrains connections but does not determine membrane properties,
receptor identity, sensory current, synaptic efficacy, timing or behavior.
Unknown transmission is explicitly approximated; no plasticity is implemented.
Sensory and output mapping predicates can be inspected and may fail if the dataset
lacks the required annotations. No fallback random neurons are substituted.
