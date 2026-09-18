# Third-party dependencies and reference projects

No external simulator source, assets or real dataset files are bundled.

Runtime dependencies: NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), Apache Arrow /
PyArrow (Apache-2.0). Optional server: FastAPI (MIT), Uvicorn (BSD-3-Clause), with
their transitive dependencies. Optional media: Pillow (HPND), OpenCV Python
packaging (MIT; OpenCV binaries under Apache-2.0 for recent versions). Optional
plotting: Matplotlib (Matplotlib/PSF-compatible license). Development: pytest
(MIT), HTTPX (BSD-3-Clause). Installed distributions include their own notices;
those notices govern the installed version.

Reference-only projects and verified source licenses are listed in
[research notes](docs/RESEARCH.md). Scientific/data citations and component terms
are in [dataset documentation](docs/DATASETS.md) and runtime manifests. The
FlyBrain license does not relicense any downloaded data or third-party component.
