"""Small semantic API with inspectable annotation-derived populations."""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections.abc import Mapping
import csv
import json
import re
import numpy as np
from .data import Connectome, load
from .dynamics import LIF, LIFParameters


class MappingUnavailable(ValueError):
    pass


ASSUMPTIONS = [
    "Uniform LIF parameters are assumed, including for naturally graded visual cells.",
    "Synapse counts are multiplied by synapse_gain_mv; this is not measured efficacy.",
    "GABA and glutamate are inhibitory; all other/unknown transmitters are excitatory proxies; receptor effects are not modeled.",
    "Inputs are sustained normalized currents until overwritten or sense.clear()/reset().",
    "No plasticity, calibrated behavior, consciousness, or validated whole-animal dynamics is claimed.",
]


@dataclass(frozen=True)
class MotorOutput:
    forward: float
    turn_left: float
    turn_right: float
    units: str = "dimensionless population-rate proxy"
    modeled: bool = True
    interpretation: str = "Mean descending rate / 100 Hz and signed bilateral differences; not measured movement."


def population(neuron, kind):
    cell_type = neuron.get("type", "")
    labels = " ".join(neuron.get(k, "") for k in ("class", "superclass", "region")).lower()
    if kind == "vision":
        return bool(re.match(r"^R[1-8](?:$|[^0-9])", cell_type)) or "photoreceptor" in labels
    if kind == "visual":
        return population(neuron, "vision") or "visual" in labels or "optic" in labels or "ol_intrinsic" in labels
    if kind == "smell":
        return "olfactory" in labels and ("sensory" in labels or cell_type.startswith("ORN")) or cell_type.startswith("ORN")
    if kind == "touch":
        return "mechanosensory" in labels or cell_type.startswith("BMN") or "bristle" in labels
    if kind == "descending":
        return "descending" in labels or cell_type.startswith("DN")
    return kind == neuron.get("region")


class Senses:
    def __init__(self, brain):
        self.brain = brain
        self.levels = {}

    def _set(self, changes):
        for key, value in changes.items():
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Intensity must be finite and between 0 and 1")
            self.brain._mapping(key, require=True)
        self.levels.update(changes)

    def vision(self, *, left=0.0, right=0.0):
        self._set({"vision.left": float(left), "vision.right": float(right)})

    def smell(self, *, channel="attractive", intensity=0.0):
        if channel not in ("attractive", "all"):
            raise ValueError("Supported smell channels: 'all', 'attractive' (uncalibrated alias)")
        # Both aliases target one state slot; calling with zero must clear either alias.
        self._set({"smell.all": float(intensity)})

    def touch(self, *, side="left", intensity=0.0):
        if side not in ("left", "right"):
            raise ValueError("Touch side must be left or right")
        self._set({f"touch.{side}": float(intensity)})

    def clear(self):
        self.levels.clear()

    def vector(self):
        result = np.zeros(len(self.brain.connectome.neurons), dtype=np.float32)
        for key, value in self.levels.items():
            result[self.brain._mapping(key)] += value
        return np.clip(result, 0, 1)


class FlyBrain:
    def __init__(self, connectome="malecns", dynamics="lif", *, parameters=None, seed=0,
                 cache_dir=None, device="cpu"):
        if device != "cpu":
            raise ValueError("This release supports device='cpu'; CUDA is not implemented")
        self.connectome = connectome if isinstance(connectome, Connectome) else load(connectome, cache_dir)
        factory = LIF if dynamics == "lif" else dynamics
        if not callable(factory):
            raise ValueError("dynamics must be 'lif' or a factory(graph, parameters, seed)")
        self.engine = factory(self.connectome, parameters or LIFParameters(), seed)
        self.parameters = self.engine.parameters
        self.sense = Senses(self)
        self._maps = {}

    @property
    def provenance(self):
        return {"connectome": self.connectome.provenance,
                "dynamics": type(self.engine).__name__, "parameters": asdict(self.parameters),
                "assumptions": ASSUMPTIONS, "mapping_version": "annotation-predicates-v1"}

    def _mapping(self, key, require=False):
        supported = {"vision.left", "vision.right", "smell.all", "smell.attractive", "touch.left", "touch.right"}
        if key not in supported:
            raise ValueError(f"Unknown input {key!r}; choose {sorted(supported)}")
        if key not in self._maps:
            kind, side = key.split(".")
            self._maps[key] = np.array([i for i, n in enumerate(self.connectome.neurons)
                if population(n, kind) and (kind == "smell" or n.get("side") == side)], dtype=np.int64)
        indices = self._maps[key]
        if require and not len(indices):
            raise MappingUnavailable(f"No annotated neurons for {key}. Inspect explain_input(), use another supported modality, or supply a custom Connectome with reviewed annotations.")
        return indices

    def explain_input(self, key):
        indices = self._mapping(key)
        kind = key.split(".")[0]
        interpretations = {
            "vision": "Bilateral photoreceptor population; frame halves proxy left/right eye illumination.",
            "smell": "Broad olfactory sensory population; 'attractive' is an uncalibrated alias for all, not established odor valence.",
            "touch": "Annotated mechanosensory/bristle population on the requested side; not a body-location map.",
        }
        return {"input": key, "available": bool(len(indices)), "count": len(indices),
                "mapped_neurons": [self.connectome.neurons[i]["id"] for i in indices],
                "dataset_annotations": [self.connectome.neurons[i] for i in indices],
                "biological_interpretation": interpretations[kind],
                "selector": {"vision": "type R1..R8 (optional non-digit suffix) or photoreceptor class; annotated side",
                             "smell": "ORN type or olfactory sensory class",
                             "touch": "mechanosensory/bristle class or BMN type; annotated side"}[kind],
                "assumptions": ASSUMPTIONS + [interpretations[kind]],
                "source": self.connectome.provenance.get("source"),
                "citations": self.connectome.provenance.get("citations", [])}

    def vision(self, frame):
        original = np.asarray(frame)
        if original.ndim not in (2, 3) or original.shape[0] < 1 or original.shape[1] < 2:
            raise ValueError("Frame must be HxW grayscale or HxWx3/4 RGB, width >= 2")
        if original.ndim == 3 and original.shape[2] not in (3, 4):
            raise ValueError("Expected RGB or RGBA channels")
        values = original.astype(np.float64)
        if original.dtype == np.uint8:
            values /= 255
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("Frame must be uint8 [0,255] or numeric [0,1]")
        if values.ndim == 3:
            values = values[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
        middle = values.shape[1] // 2
        self.sense.vision(left=float(values[:, :middle].mean()), right=float(values[:, middle:].mean()))

    def stimulus(self, values):
        """Apply a batch atomically: {'vision': {'left': 1}, 'touch': {...}}."""
        if not isinstance(values, Mapping):
            raise ValueError("Stimulus must be an object")
        previous = self.sense.levels.copy()
        try:
            for modality, kwargs in values.items():
                if modality not in ("vision", "smell", "touch") or not isinstance(kwargs, Mapping):
                    raise ValueError("Use vision/smell/touch with keyword argument objects")
                getattr(self.sense, modality)(**kwargs)
        except (ValueError, TypeError):
            self.sense.levels = previous
            raise

    def reset(self):
        self.engine.reset()
        self.sense.clear()

    def step(self, ms=10):
        self.engine.advance(self.sense.vector(), ms)
        return self

    def run(self, ms=1000):
        return self.step(ms)

    def activity(self, region=None, neurons=None):
        """Per-neuron spikes, voltage and cumulative firing rates since reset."""
        if region is not None and neurons is not None:
            raise ValueError("Select either region or neurons")
        if neurons is not None:
            try: indices = np.array([self.connectome.index[str(n)] for n in neurons], dtype=int)
            except KeyError as e: raise ValueError(f"Unknown neuron ID {e.args[0]}") from None
        elif region is not None:
            indices = np.array([i for i, n in enumerate(self.connectome.neurons) if population(n, region)], dtype=int)
            if not len(indices): raise MappingUnavailable(f"No annotated neurons in region {region!r}")
        else:
            indices = np.arange(len(self.connectome.neurons))
        return {"time_ms": self.engine.time_ms, "window": "since reset", "modeled": True,
                "neuron_ids": [self.connectome.neurons[i]["id"] for i in indices],
                "spikes": self.engine.spikes[indices].tolist(),
                "spike_counts": self.engine.counts[indices].tolist(),
                "voltage_mv": self.engine.voltage[indices].tolist(),
                "rates_hz": (self.engine.counts[indices] * 1000 / max(self.engine.time_ms, 1e-12)).tolist()}

    def output(self, name="descending"):
        if name == "motor": return asdict(self.motor())
        if name != "descending": raise ValueError("Output must be descending or motor")
        indices = [i for i, n in enumerate(self.connectome.neurons) if population(n, "descending")]
        if not indices: raise MappingUnavailable("No descending neurons annotated in this dataset")
        rates = self.engine.counts * 1000 / max(self.engine.time_ms, 1e-12)
        def mean(side):
            selected = [i for i in indices if self.connectome.neurons[i].get("side") == side]
            return float(rates[selected].mean()) if selected else None
        return {"population": "descending", "time_ms": self.engine.time_ms,
                "window": "since reset", "units": "Hz", "modeled": True,
                "count": len(indices), "mean_rate_hz": float(rates[indices].mean()),
                "left_rate_hz": mean("left"), "right_rate_hz": mean("right"),
                "interpretation": "Activity of annotation-derived descending population, not an action command.",
                "source": self.connectome.provenance.get("source")}

    def motor(self):
        output = self.output("descending")
        left, right = output["left_rate_hz"], output["right_rate_hz"]
        if left is None or right is None:
            raise MappingUnavailable("Motor proxies require descending annotations on both sides")
        return MotorOutput(float(np.clip(output["mean_rate_hz"] / 100, 0, 1)),
                           float(np.clip((left-right) / 100, 0, 1)),
                           float(np.clip((right-left) / 100, 0, 1)))

    def explain_output(self, name="descending"):
        if name not in ("descending", "motor"): raise ValueError("Unknown output")
        neurons = [n for n in self.connectome.neurons if population(n, "descending")]
        return {"output": name, "mapped_neurons": [n["id"] for n in neurons],
                "dataset_annotations": neurons, "selector": "descending class/superclass or DN type prefix",
                "biological_interpretation": "Descending neurons link brain activity to downstream circuits.",
                "assumptions": ASSUMPTIONS + [MotorOutput(0, 0, 0).interpretation],
                "source": self.connectome.provenance.get("source"),
                "citations": self.connectome.provenance.get("citations", [])}

    def log_activity(self, path, **selection):
        snapshot = self.activity(**selection)
        path = Path(path)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["neuron_id", "voltage_mv", "spike_count", "rate_hz"])
            writer.writerows(zip(snapshot["neuron_ids"], snapshot["voltage_mv"],
                                 snapshot["spike_counts"], snapshot["rates_hz"]))
        path.with_suffix(path.suffix + ".provenance.json").write_text(
            json.dumps(dict(self.provenance, time_ms=snapshot["time_ms"], window=snapshot["window"]), indent=2), encoding="utf-8")
        return path

    def visualize(self, *, top=30):
        """Optional matplotlib activity snapshot; returns Figure, no 3D/morphology claim."""
        if not isinstance(top, int) or top <= 0: raise ValueError("top must be a positive integer")
        try: import matplotlib.pyplot as plt
        except ImportError: raise ImportError("Install plotting: pip install -e '.[plot]'") from None
        indices = np.argsort(self.engine.counts)[-top:][::-1]
        snapshot = self.activity(neurons=[self.connectome.neurons[i]["id"] for i in indices])
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(snapshot["neuron_ids"], snapshot["rates_hz"])
        ax.set(ylabel="Modeled rate (Hz), since reset", xlabel="Neuron ID",
               title=f"FlyBrain activity at {self.engine.time_ms:g} ms")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        return fig
