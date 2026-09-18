from flycoder.tools.dependencies import (
    build_dependency_graph,
    extract_imports,
    find_dependencies,
)
from flycoder.tools.filesystem import Workspace


def test_extract_imports():
    content = """
import os
import flycoder.state
from flycoder.actions.registry import ActionResult
"""

    imports = extract_imports(content)

    assert "os" in imports
    assert "flycoder.state" in imports
    assert "flycoder.actions.registry" in imports


def test_find_dependencies():
    workspace = Workspace(".")
    files = workspace.list_files()

    content = """
from flycoder.state import CodingState
from flycoder.tools.filesystem import Workspace
"""

    dependencies = find_dependencies(
        content,
        files,
    )

    assert "flycoder/state.py" in dependencies
    assert (
        "flycoder/tools/filesystem.py"
        in dependencies
    )


def test_build_dependency_graph():
    workspace = Workspace(".")
    files = workspace.list_files()

    graph = build_dependency_graph(
        workspace,
        files,
        ["flycoder/agent.py"],
        max_depth=3,
    )

    assert "flycoder/agent.py" in graph

    dependencies = graph[
        "flycoder/agent.py"
    ]

    assert (
        "flycoder/state.py"
        in dependencies
    )

    assert (
        "flycoder/actions/registry.py"
        in dependencies
    )
