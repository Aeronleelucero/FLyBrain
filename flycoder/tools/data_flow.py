"""
FLY-CODER Data-Flow Intelligence.

Phase 4.3:
- Detect value flow between variables.
- Track assignments and reads.
- Track function-call inputs and outputs.
- Track returns.
- Track augmented assignments.
- Track tuple/list unpacking.
- Track containing functions and methods.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class DataFlowInfo:
    """A directed value-flow relationship."""

    source: str
    target: str
    kind: str
    file: str
    line: int
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


def _expression_name(node: ast.AST) -> str | None:
    """Return a useful name for an expression."""

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return _attribute_name(node)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return f"{node.func.id}()"

        if isinstance(node.func, ast.Attribute):
            return f"{_attribute_name(node.func)}()"

    if isinstance(node, ast.Constant):
        return repr(node.value)

    return None


def _names_in_expression(node: ast.AST) -> list[str]:
    """Return names and attributes read by an expression."""

    names: list[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Attribute):
            if isinstance(child.ctx, ast.Load):
                names.append(_attribute_name(child))

        elif isinstance(child, ast.Name):
            if isinstance(child.ctx, ast.Load):
                names.append(child.id)

    return _unique(names)


def _unique(values: list[str]) -> list[str]:
    """Return values without duplicates while preserving order."""

    result: list[str] = []

    for value in values:
        if value not in result:
            result.append(value)

    return result


def _assignment_targets(node: ast.AST) -> list[str]:
    """Return names assigned by a target expression."""

    targets: list[str] = []

    if isinstance(node, ast.Name):
        targets.append(node.id)

    elif isinstance(node, ast.Attribute):
        targets.append(_attribute_name(node))

    elif isinstance(node, (ast.Tuple, ast.List)):
        for element in node.elts:
            targets.extend(_assignment_targets(element))

    return targets


def _call_name(node: ast.Call) -> str | None:
    """Return the called function's name."""

    if isinstance(node.func, ast.Name):
        return node.func.id

    if isinstance(node.func, ast.Attribute):
        return _attribute_name(node.func)

    return None


def _call_input_sources(node: ast.Call) -> list[str]:
    """Return values supplied to a function call."""

    sources: list[str] = []

    if isinstance(node.func, ast.Attribute):
        receiver = node.func.value

        receiver_name = _expression_name(receiver)

        if receiver_name is not None:
            sources.append(receiver_name)

    for argument in node.args:
        sources.extend(_names_in_expression(argument))

    for keyword in node.keywords:
        sources.extend(_names_in_expression(keyword.value))

    return _unique(sources)


class _DataFlowVisitor(ast.NodeVisitor):
    """AST visitor that builds data-flow relationships."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.flows: list[DataFlowInfo] = []
        self._parents: list[str] = []

    @property
    def parent(self) -> str | None:
        """Return the current function or method."""

        if not self._parents:
            return None

        return self._parents[-1]

    def _add(
        self,
        source: str,
        target: str,
        kind: str,
        node: ast.AST,
    ) -> None:
        if source == target and kind == "assignment":
            return

        self.flows.append(
            DataFlowInfo(
                source=source,
                target=target,
                kind=kind,
                file=self.file_path,
                line=getattr(node, "lineno", 0),
                parent=self.parent,
            )
        )

    def _add_parameters(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        """Record function parameters as input flows."""

        positional = [
            *node.args.posonlyargs,
            *node.args.args,
        ]

        for argument in positional:
            self._add(
                source="<input>",
                target=argument.arg,
                kind="parameter",
                node=argument,
            )

        if node.args.vararg is not None:
            self._add(
                source="<input>",
                target=node.args.vararg.arg,
                kind="parameter",
                node=node.args.vararg,
            )

        for argument in node.args.kwonlyargs:
            self._add(
                source="<input>",
                target=argument.arg,
                kind="parameter",
                node=argument,
            )

        if node.args.kwarg is not None:
            self._add(
                source="<input>",
                target=node.args.kwarg.arg,
                kind="parameter",
                node=node.args.kwarg,
            )

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:
        self._parents.append(node.name)

        self._add_parameters(node)

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

        self._add_parameters(node)

        for default in node.args.defaults:
            self.visit(default)

        for default in node.args.kw_defaults:
            if default is not None:
                self.visit(default)

        for statement in node.body:
            self.visit(statement)

        self._parents.pop()

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:
        for statement in node.body:
            self.visit(statement)

    def visit_Assign(
        self,
        node: ast.Assign,
    ) -> None:
        targets: list[str] = []

        for target in node.targets:
            targets.extend(_assignment_targets(target))

        call_name = (
            _call_name(node.value)
            if isinstance(node.value, ast.Call)
            else None
        )

        if call_name is not None:
            for target in targets:
                self._add(
                    source=call_name + "()",
                    target=target,
                    kind="call_result",
                    node=node,
                )

        sources = _names_in_expression(node.value)

        for target in targets:
            for source in sources:
                self._add(
                    source=source,
                    target=target,
                    kind="assignment",
                    node=node,
                )

        self.visit(node.value)

    def visit_AnnAssign(
        self,
        node: ast.AnnAssign,
    ) -> None:
        if node.value is None:
            return

        targets = _assignment_targets(node.target)

        call_name = (
            _call_name(node.value)
            if isinstance(node.value, ast.Call)
            else None
        )

        if call_name is not None:
            for target in targets:
                self._add(
                    source=call_name + "()",
                    target=target,
                    kind="call_result",
                    node=node,
                )

        sources = _names_in_expression(node.value)

        for target in targets:
            for source in sources:
                self._add(
                    source=source,
                    target=target,
                    kind="assignment",
                    node=node,
                )

        self.visit(node.value)

    def visit_AugAssign(
        self,
        node: ast.AugAssign,
    ) -> None:
        targets = _assignment_targets(node.target)
        sources = _names_in_expression(node.value)

        for target in targets:
            self._add(
                source=target,
                target=target,
                kind="update",
                node=node,
            )

            for source in sources:
                self._add(
                    source=source,
                    target=target,
                    kind="update",
                    node=node,
                )

        self.visit(node.value)

    def visit_NamedExpr(
        self,
        node: ast.NamedExpr,
    ) -> None:
        targets = _assignment_targets(node.target)
        sources = _names_in_expression(node.value)

        call_name = (
            _call_name(node.value)
            if isinstance(node.value, ast.Call)
            else None
        )

        for target in targets:
            if call_name is not None:
                self._add(
                    source=call_name + "()",
                    target=target,
                    kind="call_result",
                    node=node,
                )

            for source in sources:
                self._add(
                    source=source,
                    target=target,
                    kind="assignment",
                    node=node,
                )

        self.visit(node.value)

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:
        call_name = _call_name(node)

        if call_name is not None:
            for source in _call_input_sources(node):
                self._add(
                    source=source,
                    target=f"{call_name}()",
                    kind="call_input",
                    node=node,
                )

        self.generic_visit(node)

    def visit_Return(
        self,
        node: ast.Return,
    ) -> None:
        if node.value is not None:
            sources = _names_in_expression(node.value)

            for source in sources:
                self._add(
                    source=source,
                    target="<return>",
                    kind="return",
                    node=node,
                )

        self.generic_visit(node)


def analyze_data_flow(
    content: str,
    file_path: str = "<memory>",
) -> list[DataFlowInfo]:
    """
    Analyze Python source code and return data-flow relationships.

    Invalid Python returns an empty list instead of raising SyntaxError.
    """

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    visitor = _DataFlowVisitor(file_path)
    visitor.visit(tree)

    return visitor.flows


def find_data_flow_by_kind(
    flows: list[DataFlowInfo],
    kind: str,
) -> list[DataFlowInfo]:
    """Return data-flow relationships matching a kind."""

    return [
        flow
        for flow in flows
        if flow.kind == kind
    ]


def find_data_flow_by_source(
    flows: list[DataFlowInfo],
    source: str,
) -> list[DataFlowInfo]:
    """Return relationships originating from a source."""

    return [
        flow
        for flow in flows
        if flow.source == source
    ]


def find_data_flow_by_target(
    flows: list[DataFlowInfo],
    target: str,
) -> list[DataFlowInfo]:
    """Return relationships ending at a target."""

    return [
        flow
        for flow in flows
        if flow.target == target
    ]


def find_data_flow_for_parent(
    flows: list[DataFlowInfo],
    parent: str,
) -> list[DataFlowInfo]:
    """Return relationships belonging to a function or method."""

    return [
        flow
        for flow in flows
        if flow.parent == parent
    ]


def build_data_flow_report(
    flows: list[DataFlowInfo],
) -> str:
    """Build a readable data-flow intelligence report."""

    lines: list[str] = [
        "FLY-CODER DATA-FLOW REPORT",
        "==========================",
    ]

    if not flows:
        lines.append("No data-flow relationships detected.")
        return "\n".join(lines)

    files = sorted({flow.file for flow in flows})

    for file_path in files:
        file_flows = [
            flow
            for flow in flows
            if flow.file == file_path
        ]

        lines.append("")
        lines.append(f"File: {file_path}")
        lines.append(f"Flows: {len(file_flows)}")

        kinds: dict[str, int] = {}

        for flow in file_flows:
            kinds[flow.kind] = kinds.get(
                flow.kind,
                0,
            ) + 1

        lines.append("Kinds:")

        for kind in sorted(kinds):
            lines.append(f"  {kind}: {kinds[kind]}")

        lines.append("Relationships:")

        for flow in file_flows:
            parent = flow.parent or "<module>"

            lines.append(
                f"  {flow.source} "
                f"--{flow.kind}--> "
                f"{flow.target} "
                f"line={flow.line} "
                f"parent={parent}"
            )

    return "\n".join(lines)