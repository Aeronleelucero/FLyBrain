"""Live CLI + TCP + JSON smoke test, bounded lifetime, synthetic data only."""
import json
import socket
import subprocess
import sys
import time
import urllib.request

with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
process = subprocess.Popen([sys.executable, "-m", "flybrain", "serve", "--dataset", "synthetic", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
base = f"http://127.0.0.1:{port}"
try:
    for _ in range(100):
        try:
            with urllib.request.urlopen(base+"/output", timeout=1) as response:
                if response.status == 200: break
        except OSError:
            if process.poll() is not None: raise RuntimeError(process.stderr.read())
            time.sleep(.1)
    else: raise RuntimeError("Server startup timed out")
    for endpoint, body in [("/stimulus", {"vision":{"left":1}}), ("/step", {"ms":100})]:
        request = urllib.request.Request(base+endpoint, data=json.dumps(body).encode(),
                                         headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
    with urllib.request.urlopen(base+"/output", timeout=5) as response:
        result = json.load(response)
    assert result["left_rate_hz"] > 0 and result["right_rate_hz"] == 0
    print(json.dumps({"live_server":"passed","output":result}, indent=2))
finally:
    process.terminate()
    try: process.wait(timeout=10)
    except subprocess.TimeoutExpired: process.kill(); process.wait()
