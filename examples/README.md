# Integration examples

All examples default to clearly labeled **synthetic** data for offline discovery.
Install optional media dependencies with `pip install -e '.[media]'`.

| Example | Run from repository root |
| --- | --- |
| Hello | `flybrain run examples/hello_brain.py` |
| Image | `flybrain run examples/image/main.py photo.png --dataset malecns` |
| Video | `flybrain run examples/video/main.py clip.mp4 --dataset malecns` |
| Webcam | `flybrain run examples/webcam/main.py --dataset malecns --frames 100` |
| Closed-loop point world | `flybrain run examples/simple_2d_world/main.py` |
| Experimental arbitrary numeric stream | `flybrain run examples/market/main.py` |

Pull the chosen real dataset first. Frame brightness is a bilateral proxy, not a
retinotopic retina. Video/webcam advance 10 simulated ms per processed frame;
they do not claim wall-clock real time. The world adapter assigns movement in
application code. The market example makes no investment or prediction claim.

See [JavaScript example](../docs/API.md#javascript) for the local HTTP interface.
For a Bad Apple demonstration, pass your own local video to the video example;
no video assets are redistributed.
