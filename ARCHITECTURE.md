# FlyBrain architecture

Status: implementation blueprint, 2026-09-11.

Environment → application adapter → semantic stimulus → immutable connectome →
replaceable dynamics → biological population activity → application adapter.

## Decisions

- Official MaleCNS v1.0 flat Feather files are the primary backend. Download from
  Janelia's public GCS bucket, without neuPrint authentication. Preserve original
  annotations, use annotated neurons, report excluded segment edges explicitly.
- FlyWire FAFB v783 uses official Zenodo connectivity (CC-BY-4.0 on that deposit)
  and pinned research-team annotations (CC BY-NC 4.0 public-release guidelines).
  These terms are independent of the SDK license. Never bundle datasets.
- Download streaming with upstream checksums when available, record SHA256 and
  source URLs. Publish a cache only after integrity checks and preprocessing.
- Keep neuron IDs as strings at API boundaries (JavaScript precision). Store raw
  synapse counts in a SciPy sparse matrix with rows=postsynaptic, columns=presynaptic.
  Aggregate duplicate pairs; no dense N×N allocation. Feather batches bound import
  memory; an on-disk edge spool avoids Python objects for millions of edges.
- Dynamics owns voltage, refractory state, spikes and time; topology owns none.
  CPU LIF first. A dynamics protocol permits custom implementations; CUDA is optional
  future work, not a claimed capability until verified.
- Explicit engineering assumptions: fixed-step LIF, count-to-weight gain,
  transmitter-to-sign policy, sustained sensory current, bilateral mean brightness,
  population-rate motor proxies. No consciousness, learning or behavior claim.
- Semantic maps use annotation predicates, never arbitrary fallback neuron IDs.
  Missing populations fail with an actionable error. Explanations expose selectors,
  matched IDs and annotations, sources, biological interpretation and assumptions.
- Local REST server binds loopback, validates inputs, serializes access to one brain.
  Application actions (BUY/FIRE/JUMP) belong exclusively in adapters.

## Modules

`data`: registry, download verification, canonical graph and cache.
`dynamics`: LIF parameters and CPU engine.
`brain`: semantic maps, sensory facade, output summaries and public API.
`adapters`: encode/decode protocol and runner.
`server`, `cli`: local integration and discoverability.

## Validation

Offline synthetic fixtures cover graph integrity, dynamics analytic behavior,
determinism, mapping, adapters, CLI and REST. Real downloads are explicit integration
checks, excluded from CI. Benchmark reports neuron/edge counts, simulated duration,
wall time and sparse storage, without extrapolating synthetic timings to whole brains.
