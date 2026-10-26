"""ESA SNAP adapter: runs the `gpt` command-line tool on a graph written by `geocrop.s1.write_snap_graph`.

Kept out of `geocrop.s1` so that the Sentinel-1 core (graph generation, import, compositing) never starts
external processes; see docs/adr/0001-layered-architecture.md.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def run_snap(gpt: str, graph: str | Path, product: str | Path, output: str | Path, cache: str = "8G",
             threads: int | None = None) -> int:
    graph = Path(graph)
    if not graph.is_file():
        raise FileNotFoundError(graph)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    command = [str(gpt), str(graph), f"-Pinput={product}", f"-Poutput={output}", "-c", cache]
    if threads:
        command += ["-q", str(threads)]
    return subprocess.run(command, check=True).returncode
