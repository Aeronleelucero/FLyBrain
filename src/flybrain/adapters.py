"""Application semantics live here, outside neuron mappings."""
from typing import Protocol, Any, Mapping


class FlyBrainAdapter(Protocol):
    def encode(self, environment: Any) -> Mapping[str, dict]:
        """Return e.g. {'vision': {'left': 1.0, 'right': 0.0}}."""
        ...

    def decode(self, brain_output: dict) -> Any:
        """Convert biological population rates to application actions."""
        ...


def interact(brain, adapter: FlyBrainAdapter, environment, ms=10):
    brain.stimulus(adapter.encode(environment))
    brain.step(ms)
    return adapter.decode(brain.output("descending"))
