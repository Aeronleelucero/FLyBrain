"""Python dependency analysis for FLY-CODER."""

from __future__ import annotations

import ast


def extract_imports(content: str) -> list[str]:
    """Extract imported module names from Python source."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports: list[str] = []

    for node in ast.walk(tree):

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):

            if node.module:
                imports.append(node.module)

    return sorted(set(imports))


def module_to_path(
    module: str,
    workspace_files: list[str],
) -> str | None:
    """Resolve a Python module name to a workspace file."""

    module_path = module.replace(".", "/")

    candidates = [
        f"{module_path}.py",
        f"{module_path}/__init__.py",
    ]

    normalized_files = {
        path.replace("\\", "/"): path
        for path in workspace_files
    }

    for candidate in candidates:

        if candidate in normalized_files:
            return normalized_files[candidate]

    return None


def find_dependencies(
    content: str,
    workspace_files: list[str],
) -> list[str]:
    """Find direct local FLY-CODER dependencies."""

    imports = extract_imports(content)

    dependencies: list[str] = []

    for module in imports:

        # Only analyze local FLY-CODER modules.
        if not module.startswith("flycoder"):
            continue

        resolved = module_to_path(
            module,
            workspace_files,
        )

        if (
            resolved
            and resolved not in dependencies
        ):
            dependencies.append(resolved)

    return dependencies


def build_dependency_graph(
    workspace,
    workspace_files: list[str],
    starting_files: list[str],
    max_depth: int = 3,
) -> dict[str, list[str]]:
    """Build a recursive local Python dependency graph."""

    graph: dict[str, list[str]] = {}

    # Prevent infinite recursion if files depend on
    # each other in a cycle.
    visited: set[str] = set()

    def visit(
        file_path: str,
        depth: int,
    ) -> None:
        """Recursively inspect one file."""

        if file_path in visited:
            return

        visited.add(file_path)

        # Stop recursion at the configured depth.
        if depth > max_depth:
            graph.setdefault(
                file_path,
                [],
            )
            return

        try:
            content = workspace.read_file(
                file_path
            )

        except (
            FileNotFoundError,
            UnicodeDecodeError,
        ):
            graph[file_path] = []
            return

        dependencies = find_dependencies(
            content,
            workspace_files,
        )

        graph[file_path] = dependencies

        # Recursively inspect dependencies.
        for dependency in dependencies:

            visit(
                dependency,
                depth + 1,
            )

    # Start from all relevant files.
    for file_path in starting_files:

        visit(
            file_path,
            0,
        )

    return graph