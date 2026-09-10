# FlyBrain integration guide

## Loop

```python
from flybrain import FlyBrain, LIFParameters

fly = FlyBrain("malecns", dynamics="lif", seed=42,
               parameters=LIFParameters(dt_ms=1, synapse_gain_mv=0.25))
fly.sense.vision(left=0.8, right=0.2)
fly.step(ms=10)
fly.run(ms=100)
print(fly.output("descending"))
print(fly.motor().forward)
```

Durations are **milliseconds**, must be nonnegative, finite and exact multiples
of `dt_ms`; invalid durations never advance state. `run` and `step` share the
same time integration. CPU float32 state and a seeded NumPy generator provide
repeatability within a fixed software/platform environment. CPU is the only
implemented device; CUDA is a future extension.

Inputs stay active until replaced, `fly.sense.clear()` or `fly.reset()`.
`sense.vision(left=1)` sets right to zero. Touch updates the requested side only.
Multiple modalities add current and clip the total to [0,1]. `reset` restores
voltage, spikes, counts, time, inputs and the initial RNG seed.

```python
fly.vision(frame)  # HxW or HxWx3/4, RGB (not OpenCV BGR)
fly.sense.smell(channel="attractive", intensity=0.7)
fly.sense.touch(side="left", intensity=1.0)
fly.explain_input("vision.left")
fly.explain_input("smell.attractive")
fly.explain_output("motor")
```

`attractive` is an **uncalibrated alias for the broad olfactory sensory population**,
not an odor-valence discovery. Use `channel="all"` for literal naming. The touch
population includes mechanosensory subclasses, without calibrated body locations.
Missing annotations raise `MappingUnavailable`; `explain_input` still returns an
empty map with `available=False`. Explanations return exact neuron IDs, original
annotations, selectors, source/citations and assumptions.

## State and output

```python
fly.activity()                         # All neurons; can be large
fly.activity(region="visual")         # Annotation-derived visual population
fly.activity(neurons=["10001"])        # IDs are strings in results
fly.output("descending")
fly.output("motor")                   # dict form of fly.motor()
fly.provenance                        # Parameters + assumptions + source manifest
fly.log_activity("activity.csv", region="visual")  # includes provenance sidecar
fig = fly.visualize(top=30)            # pip install -e '.[plot]'
fig.savefig("activity.png")
```

Activity contains current voltage (mV), last-tick spike flags, cumulative spike
counts and rates (Hz) **since reset**, not instantaneous rates. Descending output
reports mean population rates and bilateral means. Missing side information is
`None`, not an invented zero. Motor output requires both sides.

Motor proxies: `forward = clip(mean descending Hz / 100)`,
`turn_left = clip((left Hz - right Hz)/100)`, and the reverse for `turn_right`,
all clipped to [0,1]. The 100 Hz normalization and behavioral labels are engineering
choices, not validated locomotion or a measured motor command. There is no hardcoded
BUY/SELL/FIRE/JUMP neuron mapping.

## Adapter protocol

```python
from flybrain import FlyBrain, interact

class GameAdapter:
    def encode(self, game):
        return {"vision": {"left": game["left_light"], "right": game["right_light"]}}

    def decode(self, brain_output):
        # Explicit application rule; no claim about innate turning.
        left = brain_output["left_rate_hz"] or 0
        right = brain_output["right_rate_hz"] or 0
        return {"turn": (left-right)/100}

fly = FlyBrain("synthetic")
action = interact(fly, GameAdapter(), {"left_light": 1, "right_light": 0}, ms=100)
```

Plugins use structural typing: any object with `encode(environment)` and
`decode(brain_output)` satisfies `FlyBrainAdapter`. Encoding returns a mapping
of modality to keyword arguments. `interact` applies the stimulus atomically,
advances time, and passes descending output to `decode`.

For a custom connectome pass a `Connectome(neurons, sparse_counts, provenance)`.
For custom dynamics pass a factory `(connectome, parameters, seed) -> Dynamics`;
it must implement reset/advance and the state fields in `dynamics.py`. The current
facade expects LIF-compatible parameter fields. There is no automatic plugin
loading or unsafe module discovery.

## Local REST

```bash
pip install -e '.[server]'
flybrain serve --dataset malecns --port 8000
```

Default: `127.0.0.1:8000`. One brain per server process, synchronized requests.
OpenAPI: `http://127.0.0.1:8000/docs`. Use native clients or a local application
proxy; cross-origin browser requests are disabled. This is not an authenticated
public hosting service. Start another process/port for an independent brain.

| Method | Path | Request / behavior |
| --- | --- | --- |
| POST | `/stimulus` | `{"vision":{"left":1,"right":0}}` (also smell/touch) |
| POST | `/step` | `{"ms":100}`; max 1000 ms per request |
| GET | `/activity` | `region`, repeated `neurons`, `offset`, `limit` (default 1000, max 10000) |
| GET | `/output` | `name=descending` or `name=motor` |
| POST | `/reset` | Clear inputs and state |
| GET | `/provenance` | Source manifest and dynamics assumptions |
| GET | `/explain/input/vision.left` | Mapping evidence |

## JavaScript

Node.js / server-side JavaScript (or a local application proxy):

```javascript
const base = "http://127.0.0.1:8000";
async function post(path, body) {
  const response = await fetch(base + path, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
await post("/stimulus", {vision: {left: 1, right: 0}});
await post("/step", {ms: 100});
const response = await fetch(base + "/output?name=descending");
if (!response.ok) throw new Error(await response.text());
console.log(await response.json());
```

Unity, Godot, robot controllers and mods can use the same JSON endpoints through
their native HTTP client. These are protocol examples, not tested engine plugins.

## LIF equations and assumptions

At each tick of duration dt, for non-refractory neurons:

```text
v_next = v_rest + (v - v_rest) exp(-dt / tau_m)
         + stimulus_mv * input * (1 - exp(-dt / tau_m))
         + signed_count_matrix @ previous_spikes * synapse_gain_mv
```

Threshold crossings generate a spike, reset voltage and enter an integer number
of refractory ticks `ceil(refractory_ms/dt_ms)`. Presynaptic spikes arrive one
tick later. Optional noise is independent Gaussian voltage noise with standard
deviation `noise_std_mv * sqrt(dt_ms)`. There is no synaptic decay state, anatomical
conduction delay, compartmental morphology, learning or plasticity. Synapse count
gain is independent of dt because arrivals are delta voltage jumps. This simple
model is not the calibrated Shiu et al. implementation.
