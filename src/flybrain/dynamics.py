"""Independent, vectorized CPU LIF. Units: ms, mV, dimensionless stimulus."""
from dataclasses import dataclass, asdict
from typing import Protocol
import math
import numpy as np


@dataclass(frozen=True)
class LIFParameters:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    rest_mv: float = -65.0
    reset_mv: float = -65.0
    threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    synapse_gain_mv: float = 0.25
    stimulus_mv: float = 30.0
    noise_std_mv: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("LIF parameters must be finite")
        if self.dt_ms <= 0 or self.tau_m_ms <= 0 or self.dt_ms > self.tau_m_ms:
            raise ValueError("Require 0 < dt_ms <= tau_m_ms")
        if self.refractory_ms < 0 or self.noise_std_mv < 0:
            raise ValueError("Refractory period and noise must be nonnegative")
        if self.threshold_mv <= max(self.rest_mv, self.reset_mv):
            raise ValueError("Threshold must exceed rest and reset")
        if self.synapse_gain_mv < 0 or self.stimulus_mv <= 0:
            raise ValueError("Gain must be nonnegative and stimulus gain positive")


class Dynamics(Protocol):
    """A custom engine receives the same canonical Connectome as the CPU engine."""
    parameters: LIFParameters
    voltage: np.ndarray
    spikes: np.ndarray
    counts: np.ndarray
    time_ms: float

    def reset(self) -> None: ...
    def advance(self, stimulus: np.ndarray, ms: float) -> None: ...


class LIF:
    def __init__(self, connectome, parameters=None, seed=0):
        self.parameters = parameters or LIFParameters()
        self.seed = seed
        self.n = len(connectome.neurons)
        # Raw counts never change. Receptor-dependent signs are a modeling policy.
        signs = np.array([-1.0 if n.get("nt", "").upper() in
                          {"GABA", "GLUT", "GLUTAMATE"} else 1.0
                          for n in connectome.neurons], dtype=np.float32)
        self.weights = connectome.counts.astype(np.float32).multiply(signs).tocsr()
        self.weights.data *= self.parameters.synapse_gain_mv
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.voltage = np.full(self.n, self.parameters.rest_mv, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=bool)
        self.counts = np.zeros(self.n, dtype=np.int64)
        self.refractory = np.zeros(self.n, dtype=np.int64)
        self.ticks = 0
        self.time_ms = 0.0

    def advance(self, stimulus, ms):
        p = self.parameters
        if not math.isfinite(ms) or ms < 0:
            raise ValueError("Duration must be finite and nonnegative")
        steps = round(ms / p.dt_ms)
        if not math.isclose(steps * p.dt_ms, ms, abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError("Duration must be a multiple of dt_ms")
        stimulus = np.asarray(stimulus, dtype=np.float32)
        if stimulus.shape != (self.n,) or not np.isfinite(stimulus).all():
            raise ValueError("Stimulus must be a finite vector of neuron count length")
        decay = math.exp(-p.dt_ms / p.tau_m_ms)
        refractory_ticks = math.ceil(p.refractory_ms / p.dt_ms)
        for _ in range(steps):
            incoming = self.weights @ self.spikes.astype(np.float32)
            available = self.refractory == 0
            self.refractory = np.maximum(self.refractory - 1, 0)
            # Exact subthreshold leak for constant current; synapses are delta jumps
            # delivered one tick after presynaptic spikes. No plasticity.
            candidate = (p.rest_mv + (self.voltage - p.rest_mv) * decay
                         + p.stimulus_mv * stimulus * (1 - decay) + incoming)
            if p.noise_std_mv:
                candidate += self.rng.normal(0, p.noise_std_mv * math.sqrt(p.dt_ms), self.n)
            self.voltage[:] = np.where(available, candidate, p.reset_mv)
            self.spikes = available & (self.voltage >= p.threshold_mv)
            self.voltage[self.spikes] = p.reset_mv
            self.refractory[self.spikes] = refractory_ticks
            self.counts += self.spikes
            self.ticks += 1
        self.time_ms = self.ticks * p.dt_ms
