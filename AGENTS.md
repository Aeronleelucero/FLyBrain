# Integrating FlyBrain

1. `from flybrain import FlyBrain`; choose `malecns`, `flywire`, or explicit `synthetic`.
2. Run `flybrain pull malecns` once before constructing a real dataset brain.
3. Encode the environment with `fly.vision(frame)` or `fly.sense.vision/smell/touch`.
4. Advance with `fly.step(ms=10)` / `fly.run(ms=100)`; time is milliseconds.
5. Read `fly.output("descending")` or `fly.motor()` and decode in an application adapter.

Prefer semantic APIs. Do not hardcode neuron IDs unless explicitly needed.
Inspect `explain_input`, `explain_output` and `provenance` before interpreting outputs.
Inputs persist until replaced/cleared/reset. Rates are cumulative since reset.
Frames: uint8 0–255 or numeric 0–1, grayscale/RGB; convert OpenCV BGR first.
Motor fields are engineered, uncalibrated proxies. `smell.attractive` is an alias,
not biologically validated odor valence. Never claim exact biological reproduction,
consciousness, innate game behavior, demonstrated learning or financial prediction.
Keep application actions in adapters; never map BUY/FIRE/JUMP in neuron code.

# Repository work

- Read `ARCHITECTURE.md`, `docs/API.md`, `docs/DATASETS.md`.
- `pip install -e '.[dev]'`; `python -m pytest -q` runs offline fixtures only.
- `python benchmarks/benchmark.py --ms 100` is an offline benchmark.
- `python tools/real_check.py malecns` explicitly downloads ~1.1 GB, outside CI.
- Preserve sparse connectivity (post rows / pre columns); no dense N×N matrices.
- Keep dynamics separate from raw topology; retain data licenses and source hashes.
- Never silently substitute synthetic data for failed real data loading.
- Code is MIT. Dataset licenses remain independent. No third-party source vendoring.
- PyPI publishing, CUDA, calibrated retina/motor models and 3D are not implemented.
