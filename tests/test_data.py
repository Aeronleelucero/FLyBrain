import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pytest
from flybrain.data import build_graph, download, read_neurons, load, pull, REGISTRY
from flybrain.data import normalize_neuron


@pytest.mark.parametrize("dataset", ["malecns", "flywire"])
def test_official_schema_batches_aggregation_and_unknown_segments(tmp_path, dataset):
    if dataset == "malecns":
        a = tmp_path/"annotations.feather"
        feather.write_feather(pa.table({"bodyId":[1,2,3], "type":["R1","DNp01","ORN"],
                                        "somaSide":["L","R","L"]}), a)
        columns = {"body_pre":[1,1,999], "body_post":[2,2,1], "weight":[3,7,10]}
    else:
        a = tmp_path/"annotations.tsv"
        a.write_text("root_id\tcell_type\tside\n1\tR1\tleft\n2\tDNp01\tright\n3\tORN\tleft\n")
        columns = {"pre_pt_root_id":[1,1,999], "post_pt_root_id":[2,2,1], "syn_count":[3,7,10]}
    e=tmp_path/"edges.feather"
    feather.write_feather(pa.table(columns), e, chunksize=1)
    neurons = read_neurons(a,dataset)
    graph=build_graph(neurons,e,{},tmp_path)
    assert graph.counts.nnz == 1 and graph.counts[1,0] == 10
    assert graph.counts.shape == (3,3)  # retain isolated annotated neuron
    assert graph.provenance["preprocessing"]["excluded_edge_rows"] == 1
    assert graph.provenance["preprocessing"]["excluded_synapses"] == 10
    assert neurons[0]["side"] == "left"


def test_download_checksum_and_repair(tmp_path):
    source = tmp_path/"source"; source.write_bytes(b"official fixture")
    target = tmp_path/"download"
    expected="sha256:"+hashlib.sha256(source.read_bytes()).hexdigest()
    metadata=download(source.as_uri(),target,expected)
    assert metadata["sha256"] == expected.split(":")[1]
    target.write_bytes(b"corrupted")
    download(source.as_uri(),target,expected)
    assert target.read_bytes() == source.read_bytes()
    with pytest.raises(ValueError): download(source.as_uri(),tmp_path/"bad", "sha256:000")
    assert not (tmp_path/"bad").exists()


def test_pull_cache_and_corruption(tmp_path, monkeypatch):
    source = tmp_path/"source"; source.mkdir()
    feather.write_feather(pa.table({"bodyId":[1,2,3], "type":["R1","DN",None], "superclass":["ol_sensory","descending_neuron",None],
                                  "somaSide":["L","R",None]}), source/"annotations.feather")
    feather.write_feather(pa.table({"body":[1,2], "predicted_nt":["gaba","acetylcholine"]}), source/"transmitters.feather")
    feather.write_feather(pa.table({"body_pre":[1], "body_post":[2], "weight":[5]}), source/"connections.feather")
    monkeypatch.setitem(REGISTRY,"malecns", dict(REGISTRY["malecns"],files={p.name:p.as_uri() for p in source.iterdir()}))
    cache=tmp_path/"cache"
    pull("malecns",cache)
    graph=load("malecns",cache)
    assert graph.neurons[0]["nt"] == "gaba"
    assert graph.counts[1,0] == 5
    assert graph.provenance["excluded_annotation_rows"] == 1
    assert len(graph.neurons) == 2
    assert pull("malecns",cache)["dataset"] == "malecns"
    (cache/"malecns-1.0"/"neurons.json").write_text("[]")
    with pytest.raises(ValueError,match="Corrupt cache"): load("malecns",cache)


def test_missing_cache_and_bad_name(tmp_path):
    with pytest.raises(FileNotFoundError,match="flybrain pull malecns"): load("malecns",tmp_path)
    with pytest.raises(ValueError): load("../arbitrary",tmp_path)


def test_real_annotation_variants():
    male = normalize_neuron({"bodyId":11139,"type":"R1-R6","somaSide":None,
                             "rootSide":"L","superclass":"ol_sensory"},"malecns")
    assert male["side"] == "left"
    female = normalize_neuron({"root_id":"720575940628857210","top_nt":"gaba",
                               "cell_type":"R1-6","side":"right"},"flywire")
    assert female["nt"] == "gaba" and female["id"] == "720575940628857210"
