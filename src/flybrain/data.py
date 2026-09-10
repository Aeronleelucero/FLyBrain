"""Official downloads, explicit provenance, and sparse topology without dynamics."""
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
import base64
import csv
import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import datetime, timezone
import numpy as np
from scipy import sparse
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc

MALE_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATION_COMMIT = "ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0"
REGISTRY = {
    "malecns": {
        "version": "1.0", "license": "CC-BY-4.0",
        "source": "https://male-cns.janelia.org/download/",
        "citations": ["https://doi.org/10.1101/2025.10.09.680999",
                      "https://male-cns.janelia.org/"],
        "files": {
            "annotations.feather": MALE_BASE + "body-annotations-male-cns-v1.0-minconf-0.5.feather",
            "transmitters.feather": MALE_BASE + "body-neurotransmitters-male-cns-v1.0.feather",
            "connections.feather": MALE_BASE + "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
        },
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
    },
    "flywire": {
        "version": "783-annotations-2.1.0", "license": "see component licenses",
        "source": "https://zenodo.org/records/10676866",
        "component_licenses": {"connectivity": "CC-BY-4.0 (Zenodo record)",
                               "annotations": "CC-BY-NC-4.0 (FlyWire public release guidelines)"},
        "license_url": "https://flywire.ai/guidelines",
        "citations": ["https://doi.org/10.1038/s41586-024-07558-y",
                      "https://doi.org/10.1038/s41586-024-07686-5",
                      "https://doi.org/10.5281/zenodo.10676866"],
        "files": {
            "annotations.tsv": "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                + ANNOTATION_COMMIT + "/supplemental_files/Supplemental_file1_neuron_annotations.tsv",
            "connections.feather": "https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content",
        },
        "checksums": {"connections.feather": "md5:f48f972d262323a102aed49af1396b8a"},
    },
    "synthetic": {
        "version": "1", "license": "MIT", "source": "built-in test fixture",
        "citations": [], "files": {}, "synthetic": True,
    },
}


def cache_root(cache_dir=None):
    return Path(cache_dir or os.environ.get("FLYBRAIN_CACHE", Path.home() / ".cache" / "flybrain"))


def canonical_name(name):
    name = {"fafb": "flywire"}.get(name, name)
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset {name!r}. Choose: {', '.join(REGISTRY)}")
    return name


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url, target, expected=None, progress=None):
    """Atomic streamed download. Upstream MD5 validates transfer, SHA256 pins cache."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    sidecar = target.with_suffix(target.suffix + ".json")
    if target.exists() and sidecar.exists():
        meta = json.loads(sidecar.read_text())
        if meta.get("url") == url and digest(target) == meta.get("sha256"):
            if expected:
                algorithm, value = expected.split(":", 1)
                if digest(target, algorithm) != value:
                    raise ValueError(f"Checksum mismatch for {target.name}")
            return meta
    temporary = target.with_suffix(target.suffix + ".part")
    h, md5 = hashlib.sha256(), hashlib.md5()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "FlyBrain-SDK/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as stream:
            length = response.headers.get("Content-Length")
            upstream = response.headers.get("x-goog-hash", "")
            etag = response.headers.get("ETag")
            last_report = 0
            for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                stream.write(block)
                h.update(block)
                md5.update(block)
                size += len(block)
                if progress and size - last_report >= 64 * 1024 * 1024:
                    progress(f"{target.name}: {size / 1e6:.0f} MB" + (f" / {int(length) / 1e6:.0f} MB" if length else ""))
                    last_report = size
        if length and size != int(length):
            raise ValueError(f"Truncated download: {target.name}")
        verification = "HTTPS transfer and local SHA256; no upstream checksum available"
        google_md5 = next((v.strip()[4:] for v in upstream.split(",") if v.strip().startswith("md5=")), None)
        if google_md5:
            if base64.b64encode(md5.digest()).decode() != google_md5:
                raise ValueError(f"Upstream MD5 mismatch: {target.name}")
            verification = "GCS x-goog-hash MD5 verified"
        if expected:
            algorithm, value = expected.split(":", 1)
            actual = {"sha256": h.hexdigest(), "md5": md5.hexdigest()}.get(algorithm)
            if actual != value:
                raise ValueError(f"Pinned checksum mismatch: {target.name}")
            verification = f"Pinned {algorithm} verified"
        meta = {"url": url, "sha256": h.hexdigest(), "bytes": size, "etag": etag,
                "verification": verification, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        os.replace(temporary, target)
        sidecar.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class Connectome:
    """counts[post, pre] contains positive integer synapse counts, not model weights."""
    neurons: list[dict]
    counts: sparse.csr_matrix
    provenance: dict

    def __post_init__(self):
        self.neurons = [dict(n, id=str(n["id"])) for n in self.neurons]
        ids = [n["id"] for n in self.neurons]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Neuron IDs must be nonempty and unique")
        if not sparse.issparse(self.counts):
            raise ValueError("Connectivity must be sparse")
        self.counts = self.counts.tocsr(copy=True)
        self.counts.sum_duplicates()
        self.counts.eliminate_zeros()
        self.counts.sort_indices()
        values = self.counts.data
        if self.counts.shape != (len(ids), len(ids)):
            raise ValueError("Connectivity shape does not match neuron count")
        if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("Synapse counts must be finite nonnegative integers")
        self.counts.data.flags.writeable = False
        self.index = {neuron_id: i for i, neuron_id in enumerate(ids)}

    @property
    def sparse_bytes(self):
        return sum(a.nbytes for a in (self.counts.data, self.counts.indices, self.counts.indptr))


def synthetic():
    neurons = []
    for side in ("left", "right"):
        for cell_type, cell_class, region in [("R1", "visual", "visual"),
                ("ORN", "olfactory", "olfactory"), ("BMN", "mechanosensory", "sensory"),
                ("INT", "interneuron", "central"), ("DN", "descending", "descending")]:
            neurons.append({"id": str(len(neurons)+1), "type": cell_type,
                            "class": cell_class, "side": side, "region": region,
                            "nt": "ACH", "annotations": {"synthetic": True}})
    pre, post, values = [], [], []
    for offset in (0, 5):
        for a, b in [(0, 3), (1, 3), (2, 3), (3, 4)]:
            pre.append(a+offset); post.append(b+offset); values.append(100)
    graph = sparse.csr_matrix((values, (post, pre)), shape=(10, 10))
    return Connectome(neurons, graph, dict(REGISTRY["synthetic"], dataset="synthetic"))


def batches(path):
    """Read compressed Arrow IPC record batches without decompressing the full file."""
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            yield reader.get_batch(i)


def _first(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and str(value) not in ("", "nan", "None"):
            return str(value)
    return ""


def normalize_neuron(row, dataset):
    neuron_id = _first(row, "body", "bodyId", "root_id", "root_id_783", "pt_root_id")
    if not neuron_id:
        raise ValueError(f"Missing neuron ID column: {list(row)}")
    side = _first(row, "side", "rootSide", "soma_side", "somaSide").lower()
    side = {"l": "left", "r": "right"}.get(side, side)
    return {"id": neuron_id, "type": _first(row, "type", "cell_type", "hemibrain_type"),
            "class": _first(row, "class", "cell_class"),
            "superclass": _first(row, "superclass", "super_class"), "side": side,
            "nt": _first(row, "nt_type", "consensus_nt", "predicted_nt", "known_nt", "top_nt", "neurotransmitter"),
            "annotations": row}


def read_neurons(path, dataset):
    if str(path).endswith(".tsv"):
        with open(path, encoding="utf-8", newline="") as stream:
            return [normalize_neuron(row, dataset) for row in csv.DictReader(stream, delimiter="\t")]
    return [normalize_neuron(row, dataset) for batch in batches(path) for row in batch.to_pylist()]


def build_graph(neurons, edge_path, provenance, spool_dir, progress=None):
    """O(N + E) memory, disk-backed edge arrays, no dense adjacency or Python edge list."""
    ids = pa.array([int(n["id"]) for n in neurons], type=pa.uint64())
    dtype = np.dtype([("post", "<i4"), ("pre", "<i4"), ("count", "<i8")])
    total = kept = excluded = 0
    with tempfile.TemporaryDirectory(dir=spool_dir, prefix="edges-") as temporary:
        spool = Path(temporary) / "edges.bin"
        with open(spool, "wb") as stream:
            for batch in batches(edge_path):
                columns = batch.schema.names
                def column(*names):
                    for name in names:
                        if name in columns:
                            return batch.column(columns.index(name))
                    raise ValueError(f"Missing edge column {names}; found {columns}")
                pre = pc.index_in(pc.cast(column("body_pre", "pre_pt_root_id", "bodyId_pre"), pa.uint64()), value_set=ids)
                post = pc.index_in(pc.cast(column("body_post", "post_pt_root_id", "bodyId_post"), pa.uint64()), value_set=ids)
                pre = pc.fill_null(pre, -1).to_numpy(zero_copy_only=False)
                post = pc.fill_null(post, -1).to_numpy(zero_copy_only=False)
                count = column("weight", "syn_count", "synapse_count").to_numpy(zero_copy_only=False)
                if not np.isfinite(count).all() or (count <= 0).any() or (count != np.floor(count)).any():
                    raise ValueError("Invalid source synapse count")
                valid = (pre >= 0) & (post >= 0)
                records = np.empty(int(valid.sum()), dtype=dtype)
                records["pre"], records["post"], records["count"] = pre[valid], post[valid], count[valid]
                records.tofile(stream)
                total += len(pre); kept += len(records); excluded += int(count[~valid].sum())
                if progress and total % (65536 * 200) == 0:
                    progress(f"Scanned {total:,} connection rows; retained {kept:,}")
        if not kept:
            raise ValueError("No edges match annotated neurons; check dataset schemas")
        edges = np.memmap(spool, dtype=dtype, mode="r", shape=(kept,))
        graph = sparse.coo_matrix((edges["count"], (edges["post"], edges["pre"])),
                                  shape=(len(neurons), len(neurons))).tocsr()
        del edges
    provenance = dict(provenance, preprocessing={
        "scope": "induced graph of all supplied annotated neurons; unannotated segments excluded",
        "source_edge_rows": total, "retained_edge_rows": kept,
        "excluded_edge_rows": total-kept, "excluded_synapses": excluded,
        "aggregation": "sum synapse counts per directed pair across source rows/neuropils",
        "self_connections": "retained", "schema_version": 1})
    return Connectome(neurons, graph, provenance)


@contextmanager
def dataset_lock(root, name):
    root.mkdir(parents=True, exist_ok=True)
    lock = root / f"{name}.lock"
    try:
        handle = open(lock, "x")
    except FileExistsError:
        raise RuntimeError(f"Dataset operation already running. If interrupted, remove stale lock: {lock}") from None
    try:
        handle.write(str(os.getpid())); handle.close()
        yield
    finally:
        lock.unlink(missing_ok=True)


def pull(name="malecns", cache_dir=None, progress=None):
    name = canonical_name(name)
    if name == "synthetic":
        return synthetic().provenance
    spec = REGISTRY[name]
    root = cache_root(cache_dir)
    destination = root / f"{name}-{spec['version']}"
    with dataset_lock(root, name):
        if (destination / "manifest.json").exists():
            return load(name, cache_dir).provenance
        raw = root / "raw" / name
        metadata = {}
        for filename, url in spec["files"].items():
            if progress: progress(f"Fetching/verifying {filename}")
            metadata[filename] = download(url, raw / filename, spec.get("checksums", {}).get(filename), progress)
        neurons = read_neurons(raw / ("annotations.feather" if name == "malecns" else "annotations.tsv"), name)
        source_annotation_rows = len(neurons)
        if name == "malecns":
            neurons = [n for n in neurons if n["superclass"]]
        if name == "malecns":
            index = {n["id"]: n for n in neurons}
            for batch in batches(raw / "transmitters.feather"):
                for row in batch.to_pylist():
                    neuron = index.get(_first(row, "body", "bodyId"))
                    if neuron is not None:
                        neuron["nt"] = _first(row, "consensus_nt", "predicted_nt", "nt_type", "neurotransmitter")
                        neuron["transmitter_annotations"] = row
        if progress: progress(f"Preprocessing {len(neurons):,} annotated neurons")
        provenance = dict(spec, dataset=name, downloads=metadata,
                          neuron_filter="nonempty superclass" if name == "malecns" else "all released annotation rows",
                          source_annotation_rows=source_annotation_rows,
                          excluded_annotation_rows=source_annotation_rows-len(neurons))
        graph = build_graph(neurons, raw / "connections.feather", provenance, root, progress)
        with tempfile.TemporaryDirectory(dir=root, prefix=f"{name}-build-") as tmp:
            stage = Path(tmp)
            sparse.save_npz(stage / "counts.npz", graph.counts, compressed=False)
            (stage / "neurons.json").write_text(json.dumps(graph.neurons, ensure_ascii=False), encoding="utf-8")
            graph.provenance["cache_hashes"] = {f: digest(stage / f) for f in ("counts.npz", "neurons.json")}
            (stage / "manifest.json").write_text(json.dumps(graph.provenance, indent=2), encoding="utf-8")
            destination.mkdir(parents=True, exist_ok=True)
            for filename in ("counts.npz", "neurons.json", "manifest.json"):
                os.replace(stage / filename, destination / filename)
        if progress: progress(f"Ready: {len(neurons):,} neurons, {graph.counts.nnz:,} edges")
        return graph.provenance


def load(name="malecns", cache_dir=None):
    name = canonical_name(name)
    if name == "synthetic": return synthetic()
    folder = cache_root(cache_dir) / f"{name}-{REGISTRY[name]['version']}"
    if not (folder / "manifest.json").exists():
        raise FileNotFoundError(f"Dataset not cached. Run: flybrain pull {name}")
    provenance = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("counts.npz", "neurons.json"):
        if digest(folder / filename) != provenance["cache_hashes"][filename]:
            raise ValueError(f"Corrupt cache: {folder / filename}. Remove this dataset cache and pull again.")
    return Connectome(json.loads((folder / "neurons.json").read_text(encoding="utf-8")),
                      sparse.load_npz(folder / "counts.npz"), provenance)
