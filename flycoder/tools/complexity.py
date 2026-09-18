from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexityInfo:
    name: str
    kind: str
    file: str
    line: int
    end_line: int | None
    lines: int
    parameters: int
    branches: int
    loops: int
    returns: int
    exceptions: int
    calls: int
    max_nesting: int
    cyclomatic_complexity: int


def _node_end_line(node: ast.AST) -> int | None:
    return getattr(node, "end_lineno", None)


def _function_name(node: ast.AST) -> str:
    return getattr(node, "name", "<unknown>")


def _parameter_count(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    args = node.args

    return (
        len(args.posonlyargs)
        + len(args.args)
        + len(args.kwonlyargs)
        + (1 if args.vararg is not None else 0)
        + (1 if args.kwarg is not None else 0)
    )


class _ComplexityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.branches = 0
        self.loops = 0
        self.returns = 0
        self.exceptions = 0
        self.calls = 0
        self.max_nesting = 0

        self._nesting = 0

    def _enter_nesting(self) -> None:
        self._nesting += 1
        self.max_nesting = max(self.max_nesting, self._nesting)

    def _leave_nesting(self) -> None:
        self._nesting -= 1

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1

        self._enter_nesting()

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches += 1

        self._enter_nesting()
        self.visit(node.body)
        self.visit(node.test)
        self._leave_nesting()

        self.visit(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1

        self._enter_nesting()

        self.visit(node.target)
        self.visit(node.iter)

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops += 1

        self._enter_nesting()

        self.visit(node.target)
        self.visit(node.iter)

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1

        self._enter_nesting()

        self.visit(node.test)

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        self._enter_nesting()

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for handler in node.handlers:
            self.exceptions += 1

            self._enter_nesting()
            self.visit(handler)
            self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

        for statement in node.finalbody:
            self.visit(statement)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._enter_nesting()

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

        for handler in node.handlers:
            self.exceptions += 1

            self._enter_nesting()
            self.visit(handler)
            self._leave_nesting()

        for statement in node.orelse:
            self.visit(statement)

        for statement in node.finalbody:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._enter_nesting()

        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._enter_nesting()

        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self.visit(item.optional_vars)

        for statement in node.body:
            self.visit(statement)

        self._leave_nesting()

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        # Every additional boolean operand introduces another decision path.
        self.branches += max(0, len(node.values) - 1)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        self.calls += 1
        self.generic_visit(node)


def _analyze_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file: str,
    kind: str,
) -> ComplexityInfo:
    visitor = _ComplexityVisitor()

    for statement in node.body:
        visitor.visit(statement)

    end_line = _node_end_line(node)

    if end_line is None:
        lines = 1
    else:
        lines = max(1, end_line - node.lineno + 1)

    return ComplexityInfo(
        name=_function_name(node),
        kind=kind,
        file=file,
        line=node.lineno,
        end_line=end_line,
        lines=lines,
        parameters=_parameter_count(node),
        branches=visitor.branches,
        loops=visitor.loops,
        returns=visitor.returns,
        exceptions=visitor.exceptions,
        calls=visitor.calls,
        max_nesting=visitor.max_nesting,
        cyclomatic_complexity=1 + visitor.branches + visitor.loops + visitor.exceptions,
    )


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.results: list[ComplexityInfo] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.results.append(
            _analyze_function(
                node,
                self.file,
                "function",
            )
        )

        # Continue walking so nested functions are also analyzed.
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.results.append(
            _analyze_function(
                node,
                self.file,
                "async_function",
            )
        )

        self.generic_visit(node)


def analyze_complexity(source: str, file: str = "<memory>") -> list[ComplexityInfo]:
    """
    Analyze function and method complexity from Python source code.

    Returns an empty list when the source contains a syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = _FunctionCollector(file)
    collector.visit(tree)

    return collector.results


def find_complexity_by_name(
    results: list[ComplexityInfo],
    name: str,
) -> list[ComplexityInfo]:
    return [item for item in results if item.name == name]


def find_complexity_by_kind(
    results: list[ComplexityInfo],
    kind: str,
) -> list[ComplexityInfo]:
    return [item for item in results if item.kind == kind]


def build_complexity_report(
    results: list[ComplexityInfo],
) -> str:
    lines = [
        "FLY-CODER COMPLEXITY REPORT",
        "=" * 80,
        f"Functions analyzed: {len(results)}",
        "",
    ]

    if not results:
        lines.append("No functions or methods found.")
        return "\n".join(lines)

    for item in results:
        lines.extend(
            [
                f"{item.kind}: {item.name}",
                f"  File: {item.file}",
                f"  Lines: {item.line}-{item.end_line}",
                f"  Size: {item.lines} lines",
                f"  Parameters: {item.parameters}",
                f"  Branches: {item.branches}",
                f"  Loops: {item.loops}",
                f"  Returns: {item.returns}",
                f"  Exceptions: {item.exceptions}",
                f"  Calls: {item.calls}",
                f"  Max nesting: {item.max_nesting}",
                f"  Cyclomatic complexity: {item.cyclomatic_complexity}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()