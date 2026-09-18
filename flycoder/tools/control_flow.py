"""
FLY-CODER Control-Flow Intelligence.

Phase 4.1:
- Detect control-flow constructs using Python AST.
- Track containing functions/methods/classes.
- Track nesting depth.
- Produce structured control-flow nodes.
- Produce a readable control-flow report.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ControlFlowNode:
    """A single control-flow construct discovered in source code."""

    kind: str
    file: str
    line: int
    end_line: int | None = None
    parent: str | None = None
    depth: int = 0


_CONTROL_FLOW_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Return,
    ast.Break,
    ast.Continue,
    ast.Raise,
)


def _node_end_line(node: ast.AST) -> int | None:
    """Return the ending source line for an AST node when available."""
    return getattr(node, "end_lineno", None)


def _function_name(node: ast.AST) -> str | None:
    """Return the name for a function-like AST node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node.name

    return None


def _control_flow_kind(node: ast.AST) -> str | None:
    """Map an AST node to the public control-flow kind."""
    if isinstance(node, ast.If):
        return "if"

    if isinstance(node, (ast.For, ast.AsyncFor)):
        return "for"

    if isinstance(node, ast.While):
        return "while"

    if isinstance(node, ast.Try):
        return "try"

    if isinstance(node, (ast.With, ast.AsyncWith)):
        return "with"

    if isinstance(node, ast.Return):
        return "return"

    if isinstance(node, ast.Break):
        return "break"

    if isinstance(node, ast.Continue):
        return "continue"

    if isinstance(node, ast.Raise):
        return "raise"

    return None


def _walk_control_flow(
    nodes: list[ast.stmt],
    file_path: str,
    parent: str | None,
    depth: int,
    output: list[ControlFlowNode],
) -> None:
    """Recursively walk statements and collect control-flow nodes."""

    for node in nodes:
        kind = _control_flow_kind(node)

        if kind is not None:
            output.append(
                ControlFlowNode(
                    kind=kind,
                    file=file_path,
                    line=getattr(node, "lineno", 0),
                    end_line=_node_end_line(node),
                    parent=parent,
                    depth=depth,
                )
            )

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _walk_control_flow(
                node.body,
                file_path,
                node.name,
                depth,
                output,
            )
            continue

        if isinstance(node, ast.ClassDef):
            _walk_control_flow(
                node.body,
                file_path,
                parent,
                depth,
                output,
            )
            continue

        if isinstance(node, ast.If):
            _walk_control_flow(
                node.body,
                file_path,
                parent,
                depth + 1,
                output,
            )
            _walk_control_flow(
                node.orelse,
                file_path,
                parent,
                depth + 1,
                output,
            )
            continue

        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            _walk_control_flow(
                node.body,
                file_path,
                parent,
                depth + 1,
                output,
            )
            _walk_control_flow(
                node.orelse,
                file_path,
                parent,
                depth + 1,
                output,
            )
            continue

        if isinstance(node, ast.Try):
            _walk_control_flow(
                node.body,
                file_path,
                parent,
                depth + 1,
                output,
            )

            for handler in node.handlers:
                _walk_control_flow(
                    handler.body,
                    file_path,
                    parent,
                    depth + 1,
                    output,
                )

            _walk_control_flow(
                node.orelse,
                file_path,
                parent,
                depth + 1,
                output,
            )

            _walk_control_flow(
                node.finalbody,
                file_path,
                parent,
                depth + 1,
                output,
            )
            continue

        if isinstance(node, (ast.With, ast.AsyncWith)):
            _walk_control_flow(
                node.body,
                file_path,
                parent,
                depth + 1,
                output,
            )


def analyze_control_flow(
    content: str,
    file_path: str = "<memory>",
) -> list[ControlFlowNode]:
    """
    Analyze Python source code and return discovered control-flow nodes.

    Invalid Python returns an empty list instead of raising SyntaxError.
    """

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    output: list[ControlFlowNode] = []

    _walk_control_flow(
        tree.body,
        file_path,
        parent=None,
        depth=0,
        output=output,
    )

    return output


def find_control_flow_by_kind(
    nodes: list[ControlFlowNode],
    kind: str,
) -> list[ControlFlowNode]:
    """Return control-flow nodes matching a specific kind."""
    return [node for node in nodes if node.kind == kind]


def find_control_flow_for_parent(
    nodes: list[ControlFlowNode],
    parent: str,
) -> list[ControlFlowNode]:
    """Return control-flow nodes belonging to a function or method."""
    return [node for node in nodes if node.parent == parent]


def build_control_flow_report(
    nodes: list[ControlFlowNode],
) -> str:
    """Build a human-readable control-flow intelligence report."""

    lines: list[str] = [
        "FLY-CODER CONTROL-FLOW REPORT",
        "==============================",
    ]

    if not nodes:
        lines.append("No control-flow constructs detected.")
        return "\n".join(lines)

    files = sorted({node.file for node in nodes})

    for file_path in files:
        file_nodes = [
            node
            for node in nodes
            if node.file == file_path
        ]

        lines.append("")
        lines.append(f"File: {file_path}")
        lines.append(f"Nodes: {len(file_nodes)}")

        kinds: dict[str, int] = {}

        for node in file_nodes:
            kinds[node.kind] = kinds.get(node.kind, 0) + 1

        lines.append("Kinds:")

        for kind in sorted(kinds):
            lines.append(f"  {kind}: {kinds[kind]}")

        lines.append("Flow:")

        for node in file_nodes:
            parent = node.parent or "<module>"
            end_line = (
                str(node.end_line)
                if node.end_line is not None
                else "?"
            )

            lines.append(
                f"  {'  ' * node.depth}"
                f"{node.kind} "
                f"line={node.line}-{end_line} "
                f"parent={parent} "
                f"depth={node.depth}"
            )

    return "\n".join(lines)