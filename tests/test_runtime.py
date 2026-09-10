import math
import numpy as np
import pytest
from scipy import sparse
from flybrain import FlyBrain, Connectome, LIF, LIFParameters, interact, MappingUnavailable
from flybrain.data import synthetic


def test_synthetic_end_to_end_and_direction():
    brain = FlyBrain("synthetic")
    brain.sense.vision(left=1)
    brain.run(100)
    output = brain.output("descending")
    assert output["left_rate_hz"] > 0
    assert output["right_rate_hz"] == 0
    assert brain.motor().turn_left > 0
    assert brain.activity(neurons=[1])["spike_counts"][0] > 0
    assert brain.activity(neurons=[6])["spike_counts"] == [0]
    assert brain.connectome.counts[3, 0] == 100


def test_leak_matches_analytic_solution():
    graph = Connectome([{"id": "1"}], sparse.csr_matrix((1, 1)), {})
    engine = LIF(graph)
    engine.voltage[0] = -55
    engine.advance(np.zeros(1), 10)
    expected = -65 + 10 * math.exp(-10 / 20)
    assert engine.voltage[0] == pytest.approx(expected, abs=1e-4)
    assert engine.counts[0] == 0


def test_spike_reset_and_refractory():
    graph = Connectome([{"id": "1"}], sparse.csr_matrix((1, 1)), {})
    engine = LIF(graph, LIFParameters(stimulus_mv=1000))
    engine.advance(np.ones(1), 1)
    assert engine.spikes[0] and engine.voltage[0] == -65
    engine.advance(np.ones(1), 2)
    assert engine.counts[0] == 1
    engine.advance(np.ones(1), 1)
    assert engine.counts[0] == 2


def test_inhibition_preserves_topology():
    graph = Connectome([{"id":"a","nt":"GABA"},{"id":"b"}],
                       sparse.csr_matrix(([10], ([1], [0])), shape=(2,2)), {})
    engine = LIF(graph)
    engine.spikes[0] = True
    engine.advance(np.zeros(2), 1)
    assert engine.voltage[1] < -65
    assert graph.counts[1,0] == 10


def test_seed_reset_and_chunked_run_are_deterministic():
    p = LIFParameters(noise_std_mv=1)
    a, b = FlyBrain("synthetic", parameters=p, seed=7), FlyBrain("synthetic", parameters=p, seed=7)
    a.sense.vision(left=1); b.sense.vision(left=1)
    a.run(100)
    for _ in range(10): b.step(10)
    assert a.activity() == b.activity()
    saved = a.activity()
    a.reset()
    assert a.engine.time_ms == 0 and not a.sense.levels
    a.sense.vision(left=1); a.run(100)
    assert saved == a.activity()


@pytest.mark.parametrize("duration", [-1, .5, float("nan"), float("inf")])
def test_invalid_duration_is_atomic(duration):
    brain = FlyBrain("synthetic")
    with pytest.raises(ValueError): brain.run(duration)
    assert brain.engine.time_ms == 0


def test_semantic_inputs_explanations_and_images():
    brain = FlyBrain("synthetic")
    frame = np.zeros((4, 6, 3), dtype=np.uint8); frame[:, :3] = 255
    brain.vision(frame)
    assert brain.sense.levels == {"vision.left": 1, "vision.right": 0}
    explanation = brain.explain_input("vision.left")
    assert explanation["mapped_neurons"] == ["1"]
    assert explanation["dataset_annotations"] and explanation["assumptions"]
    assert "not established odor valence" in brain.explain_input("smell.attractive")["biological_interpretation"]
    brain.sense.smell(channel="attractive", intensity=.7)
    brain.sense.smell(channel="all", intensity=0)
    assert brain.sense.levels["smell.all"] == 0
    brain.sense.touch(side="left", intensity=1)
    assert brain.sense.vector()[2] == 1
    assert len(brain.activity(region="visual")["neuron_ids"]) == 2
    with pytest.raises(ValueError): brain.vision(np.ones((1,1)))
    with pytest.raises(ValueError): brain.vision(np.full((2,2), np.nan))


def test_input_batch_rolls_back_on_invalid_modality():
    brain = FlyBrain("synthetic")
    with pytest.raises(ValueError):
        brain.stimulus({"vision": {"left": 1}, "buy": {"intensity": 1}})
    assert brain.sense.levels == {}
    with pytest.raises(ValueError): brain.sense.vision(left=float("inf"))


def test_missing_mapping_is_explicit():
    graph = Connectome([{"id":"1"}], sparse.csr_matrix((1,1)), {})
    brain = FlyBrain(graph)
    assert not brain.explain_input("vision.left")["available"]
    with pytest.raises(MappingUnavailable): brain.sense.vision(left=1)
    with pytest.raises(MappingUnavailable): brain.output()
    with pytest.raises(ValueError): brain.activity(neurons=["999"])


def test_adapter_boundary():
    class WorldAdapter:
        def encode(self, environment): return {"vision": {"left": environment}}
        def decode(self, brain_output): return "LEFT" if brain_output["left_rate_hz"] > 0 else "IDLE"
    assert interact(FlyBrain("synthetic"), WorldAdapter(), 1, ms=100) == "LEFT"


def test_custom_dynamics_factory():
    called = []
    def factory(graph, parameters, seed):
        called.append(seed)
        return LIF(graph, parameters, seed)
    FlyBrain("synthetic", dynamics=factory, seed=42).run(1)
    assert called == [42]


def test_graph_integrity_and_no_dense_topology():
    with pytest.raises(ValueError): Connectome([{"id":"1"}], np.ones((1,1)), {})
    with pytest.raises(ValueError): Connectome([{"id":"1"},{"id":"1"}], sparse.eye(2), {})
    with pytest.raises(ValueError): Connectome([{"id":"1"}], sparse.csr_matrix([[-1]]), {})
    assert synthetic().sparse_bytes < 10*10*8


def test_log_provenance(tmp_path):
    import json
    brain = FlyBrain("synthetic"); brain.run(10)
    path = brain.log_activity(tmp_path/"activity.csv")
    assert path.read_text().startswith("neuron_id,")
    assert json.loads(path.with_suffix(".csv.provenance.json").read_text())["time_ms"] == 10
