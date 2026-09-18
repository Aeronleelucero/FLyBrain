"""
FLY-CODER Symbol Reference Intelligence.

Phase 4.2:
- Detect symbol/name references.
- Detect attribute references.
- Detect calls and instantiations.
- Track containing functions/methods.
- Track assignment and read contexts.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ReferenceInfo:
    """A reference to a symbol or attribute."""

    name: str
    kind: str
    file: str
    line: int
    end_line: int | None = None
    parent: str | None = None


def _attribute_name(node: ast.Attribute) -> str:
    """Return a complete dotted attribute name."""
    parts: list[str] = []
    current: ast.AST = node

    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value

    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        parts.append(ast.unparse(current))

    return ".".join(reversed(parts))


def _node_end_line(node: ast.AST) -> int | None:
    """Return the ending source line when available."""
    return getattr(node, "end_lineno", None)


def _parent_name(stack: list[str]) -> str | None:
    """Return the innermost containing function or method."""
    if not stack:
        return None

    return stack[-1]


class _ReferenceVisitor(ast.NodeVisitor):
    """AST visitor that collects symbol references."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.references: list[ReferenceInfo] = []
        self._parents: list[str] = []

    def _add(
        self,
        name: str,
        kind: str,
        node: ast.AST,
    ) -> None:
        self.references.append(
            ReferenceInfo(
                name=name,
                kind=kind,
                file=self.file_path,
                line=getattr(node, "lineno", 0),
                end_line=_node_end_line(node),
                parent=_parent_name(self._parents),
            )
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._parents.append(node.name)

        for decorator in node.decorator_list:
            self.visit(decorator)

        for default in node.args.defaults:
            self.visit(default)

        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

        for statement in node.body:
            self.visit(statement)

        self._parents.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:
        self._parents.append(node.name)

        for decorator in node.decorator_list:
            self.visit(decorator)

        for default in node.args.defaults:
            self.visit(default)

        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

        for statement in node.body:
            self.visit(statement)

        self._parents.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)

        for base in node.bases:
            self.visit(base)

        for keyword in node.keywords:
            self.visit(keyword.value)

        for statement in node.body:
            self.visit(statement)

    def visit_Call(self, node: ast.Call) -> None:
        name: str | None = None

        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = _attribute_name(node.func)

        if name is not None:
            kind = "instantiation" if name[:1].isupper() else "call"

            self._add(
                name=name,
                kind=kind,
                node=node,
            )

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            kind = "assignment"
        elif isinstance(node.ctx, ast.Del):
            kind = "delete"
        else:
            kind = "read"

        self._add(
            name=node.id,
            kind=kind,
            node=node,
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, ast.Store):
            kind = "assignment"
        elif isinstance(node.ctx, ast.Del):
            kind = "delete"
        else:
            kind = "read"

        self._add(
            name=_attribute_name(node),
            kind=kind,
            node=node,
        )

        self.generic_visit(node)


def analyze_references(
    content: str,
    file_path: str = "<memory>",
) -> list[ReferenceInfo]:
    """
    Analyze Python source code and return symbol references.

    Invalid Python returns an empty list instead of raising SyntaxError.
    """

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    visitor = _ReferenceVisitor(file_path)
    visitor.visit(tree)

    return visitor.references


def find_references_by_name(
    references: list[ReferenceInfo],
    name: str,
) -> list[ReferenceInfo]:
    """Return references matching an exact name."""
    return [
        reference
        for reference in references
        if reference.name == name
    ]


def find_references_by_kind(
    references: list[ReferenceInfo],
    kind: str,
) -> list[ReferenceInfo]:
    """Return references matching a reference kind."""
    return [
        reference
        for reference in references
        if reference.kind == kind
    ]


def find_references_for_parent(
    references: list[ReferenceInfo],
    parent: str,
) -> list[ReferenceInfo]:
    """Return references belonging to a function or method."""
    return [
        reference
        for reference in references
        if reference.parent == parent
    ]


def build_reference_report(
    references: list[ReferenceInfo],
) -> str:
    """Build a readable symbol-reference report."""

    lines: list[str] = [
        "FLY-CODER SYMBOL REFERENCE REPORT",
        "=================================",
    ]

    if not references:
        lines.append("No symbol references detected.")
        return "\n".join(lines)

    files = sorted({reference.file for reference in references})

    for file_path in files:
        file_references = [
            reference
            for reference in references
            if reference.file == file_path
        ]

        lines.append("")
        lines.append(f"File: {file_path}")
        lines.append(f"References: {len(file_references)}")

        kinds: dict[str, int] = {}

        for reference in file_references:
            kinds[reference.kind] = kinds.get(
                reference.kind,
                0,
            ) + 1

        lines.append("Kinds:")

        for kind in sorted(kinds):
            lines.append(f"  {kind}: {kinds[kind]}")

        lines.append("References:")

        for reference in file_references:
            parent = reference.parent or "<module>"

            end_line = (
                str(reference.end_line)
                if reference.end_line is not None
                else "?"
            )

            lines.append(
                f"  {reference.name} "
                f"kind={reference.kind} "
                f"line={reference.line}-{end_line} "
                f"parent={parent}"
            )

    return "\n".join(lines)