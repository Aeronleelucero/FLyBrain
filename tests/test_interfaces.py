import json
import subprocess
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from flybrain import FlyBrain
from flybrain.server import create_app


def cli(*args):
    return subprocess.run([sys.executable,"-m","flybrain",*args],capture_output=True,text=True)


def test_cli(tmp_path):
    assert cli("--help").returncode == 0
    assert "malecns" in json.loads(cli("datasets").stdout)
    info = cli("--cache-dir",str(tmp_path),"info","malecns")
    assert info.returncode == 0 and not json.loads(info.stdout)["cached"]
    assert cli("pull","synthetic").returncode == 0
    assert cli("inspect","1","--dataset","synthetic").returncode == 0
    pathway = json.loads(cli("pathway","1","5","--dataset","synthetic").stdout)
    assert pathway["path"] == ["1","4","5"]
    assert cli("run", "examples/hello_brain.py").returncode == 0
    assert cli("run", "missing.py").returncode == 2


def test_rest_loop_validation_and_pagination():
    with TestClient(create_app(FlyBrain("synthetic"))) as client:
        assert client.post("/stimulus",json={"vision":{"left":1}}).status_code == 200
        assert client.post("/step",json={"ms":100}).json() == {"time_ms":100}
        output=client.get("/output").json()
        assert output["left_rate_hz"] > 0 and output["right_rate_hz"] == 0
        activity=client.get("/activity?limit=2&offset=1").json()
        assert activity["total"] == 10 and activity["neuron_ids"] == ["2","3"]
        assert client.get("/provenance").json()["assumptions"]
        assert client.get("/explain/input/vision.left").status_code == 200
        assert client.post("/step",json={"ms":-1}).status_code == 422
        assert client.post("/step",json={"ms":.5}).status_code == 422
        assert client.post("/stimulus",json={"vision":{"left":2}}).status_code == 422
        assert client.post("/stimulus",json={"vision":{"unexpected":1}}).status_code == 422
        assert client.get("/output?name=BUY").status_code == 422
        assert client.post("/reset").status_code == 200
        assert client.get("/output").json()["mean_rate_hz"] == 0
        assert client.post("/reset",headers={"Origin":"https://example.com"}).status_code == 403
        assert client.get("/output",headers={"Host":"evil.example"}).status_code == 400
