"""The CLI executes user-selected Python scripts with ordinary Python privileges."""
import argparse
import json
import runpy
import sys
from pathlib import Path
from .data import REGISTRY, pull, load, canonical_name
from .brain import FlyBrain


def main(argv=None):
    parser = argparse.ArgumentParser(prog="flybrain", description="Plug a fruit-fly connectome into anything")
    parser.add_argument("--cache-dir", help="Override FLYBRAIN_CACHE")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("datasets", help="List versions, sources and licenses")
    for command in ("pull", "info"):
        p = sub.add_parser(command)
        p.add_argument("dataset", choices=[*REGISTRY, "fafb"])
    p = sub.add_parser("run", help="Run a local Python integration example")
    p.add_argument("script", type=Path)
    p.add_argument("args", nargs=argparse.REMAINDER)
    p = sub.add_parser("serve", help="Start loopback REST server")
    p.add_argument("--dataset", default="malecns")
    p.add_argument("--port", type=int, default=8000)
    p = sub.add_parser("inspect", help="Inspect source annotations of a neuron")
    p.add_argument("neuron")
    p.add_argument("--dataset", default="malecns")
    p = sub.add_parser("pathway", help="Find a directed unweighted shortest path")
    p.add_argument("source"); p.add_argument("target")
    p.add_argument("--dataset", default="malecns")
    args = parser.parse_args(argv)
    try:
        if args.command == "datasets":
            print(json.dumps(REGISTRY, indent=2))
        elif args.command == "pull":
            result = pull(args.dataset, args.cache_dir, progress=lambda text: print(text, file=sys.stderr, flush=True))
            print(json.dumps(result, indent=2))
        elif args.command == "info":
            name = canonical_name(args.dataset)
            try:
                graph = load(name, args.cache_dir)
                result = dict(graph.provenance, cached=True, neurons=len(graph.neurons),
                              edges=graph.counts.nnz, sparse_bytes=graph.sparse_bytes)
            except FileNotFoundError:
                result = dict(REGISTRY[name], cached=False)
            print(json.dumps(result, indent=2))
        elif args.command == "run":
            if not args.script.is_file(): raise ValueError(f"Script not found: {args.script}")
            old_argv, old_path = sys.argv, sys.path.copy()
            try:
                sys.argv = [str(args.script), *args.args]
                sys.path.insert(0, str(args.script.resolve().parent))
                runpy.run_path(str(args.script), run_name="__main__")
            finally:
                sys.argv = old_argv; sys.path[:] = old_path
        elif args.command == "serve":
            from .server import create_app
            import uvicorn
            uvicorn.run(create_app(FlyBrain(args.dataset, cache_dir=args.cache_dir)),
                        host="127.0.0.1", port=args.port)
        else:
            graph = load(args.dataset, args.cache_dir)
            if args.command == "inspect":
                if args.neuron not in graph.index: raise ValueError("Unknown neuron ID")
                print(json.dumps(graph.neurons[graph.index[args.neuron]], indent=2))
            else:
                from scipy.sparse.csgraph import shortest_path
                if args.source not in graph.index or args.target not in graph.index:
                    raise ValueError("Unknown source or target neuron")
                start, end = graph.index[args.source], graph.index[args.target]
                _, predecessors = shortest_path(graph.counts.T.tocsr(), directed=True, unweighted=True,
                                                 indices=start, return_predecessors=True)
                path = [end]
                while path[-1] != start:
                    previous = int(predecessors[path[-1]])
                    if previous < 0: path = []; break
                    path.append(previous)
                print(json.dumps({"path": [graph.neurons[i]["id"] for i in reversed(path)],
                                  "interpretation": "Topological path, not a validated functional pathway"}))
    except (ValueError, FileNotFoundError, RuntimeError, OSError, ImportError) as error:
        parser.exit(2, f"flybrain: {error}\n")
