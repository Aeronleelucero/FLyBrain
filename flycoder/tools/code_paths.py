from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class CodePath:
    name: str
    kind: str
    file: str
    start_line: int
    end_line: int | None
    path_count: int
    exit_points: int
    branches: int
    loops: int
    exceptions: int
    returns: int
    raises: int
    max_nesting: int


def _node_end_line(node: ast.AST) -> int | None:
    return getattr(node, "end_lineno", None)


class _PathVisitor(ast.NodeVisitor):
    """
    Collect control-flow information for one function.

    Nested function definitions are intentionally treated as separate
    analysis scopes and are not traversed by this visitor.
    """

    def __init__(self) -> None:
        self.branches = 0
        self.loops = 0
        self.exceptions = 0
        self.returns = 0
        self.raises = 0
        self.exit_points = 0
        self.max_nesting = 0

        self._nesting = 0

    def _enter(self) -> None:
        self._nesting += 1
        self.max_nesting = max(self.max_nesting, self._nesting)

    def _leave(self) -> None:
        self._nesting -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Nested function is a separate code-path analysis scope.
        # Do not count its control flow as part of the enclosing function.
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Nested async function is a separate code-path analysis scope.
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambda bodies are also separate executable scopes.
        return

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1

        self._enter()

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for statement in node.orelse:
            self.visit(statement)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches += 1

        self._enter()
        self.visit(node.test)
        self.visit(node.body)
        self._leave()

        self.visit(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1

        self._enter()

        self.visit(node.target)
        self.visit(node.iter)

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for statement in node.orelse:
            self.visit(statement)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops += 1

        self._enter()

        self.visit(node.target)
        self.visit(node.iter)

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for statement in node.orelse:
            self.visit(statement)

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1

        self._enter()

        self.visit(node.test)

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for statement in node.orelse:
            self.visit(statement)

    def visit_Try(self, node: ast.Try) -> None:
        self._enter()

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for handler in node.handlers:
            self.exceptions += 1

            self._enter()
            self.visit(handler)
            self._leave()

        for statement in node.orelse:
            self.visit(statement)

        for statement in node.finalbody:
            self.visit(statement)

    def visit_TryStar(self, node: ast.TryStar) -> None:
        self._enter()

        for statement in node.body:
            self.visit(statement)

        self._leave()

        for handler in node.handlers:
            self.exceptions += 1

            self._enter()
            self.visit(handler)
            self._leave()

        for statement in node.orelse:
            self.visit(statement)

        for statement in node.finalbody:
            self.visit(statement)

    def visit_With(self, node: ast.With) -> None:
        self._enter()

        for item in node.items:
            self.visit(item.context_expr)

            if item.optional_vars is not None:
                self.visit(item.optional_vars)

        for statement in node.body:
            self.visit(statement)

        self._leave()

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._enter()

        for item in node.items:
            self.visit(item.context_expr)

            if item.optional_vars is not None:
                self.visit(item.optional_vars)

        for statement in node.body:
            self.visit(statement)

        self._leave()

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.exit_points += 1

        if node.value is not None:
            self.visit(node.value)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.raises += 1
        self.exit_points += 1

        if node.exc is not None:
            self.visit(node.exc)

        if node.cause is not None:
            self.visit(node.cause)


def _estimate_path_count(visitor: _PathVisitor) -> int:
    """
    Estimate independent control-flow paths.

    Branches add alternative paths.
    Loops add a zero-iteration alternative.
    Exception handlers add alternative exception paths.

    This is a static approximation rather than a proof of reachability.
    """
    return max(
        1,
        1
        + visitor.branches
        + visitor.loops
        + visitor.exceptions,
    )


def _analyze_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    file: str,
    kind: str,
) -> CodePath:
    visitor = _PathVisitor()

    for statement in node.body:
        visitor.visit(statement)

    return CodePath(
        name=node.name,
        kind=kind,
        file=file,
        start_line=node.lineno,
        end_line=_node_end_line(node),
        path_count=_estimate_path_count(visitor),
        exit_points=visitor.exit_points,
        branches=visitor.branches,
        loops=visitor.loops,
        exceptions=visitor.exceptions,
        returns=visitor.returns,
        raises=visitor.raises,
        max_nesting=visitor.max_nesting,
    )


class _FunctionCollector(ast.NodeVisitor):
    def __init__(self, file: str) -> None:
        self.file = file
        self.results: list[CodePath] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.results.append(
            _analyze_function(
                node,
                self.file,
                "function",
            )
        )

        # Continue into the body so nested functions are discovered by
        # the collector, but each nested function gets its own analysis.
        for statement in node.body:
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                self.visit(statement)
            else:
                self._visit_nested_definitions(statement)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.results.append(
            _analyze_function(
                node,
                self.file,
                "async_function",
            )
        )

        for statement in node.body:
            if isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                self.visit(statement)
            else:
                self._visit_nested_definitions(statement)

    def _visit_nested_definitions(self, node: ast.AST) -> None:
        """
        Search a function body for nested function definitions without
        analyzing their control flow as part of the parent.
        """

        for child in ast.iter_child_nodes(node):
            if isinstance(
                child,
                (ast.FunctionDef, ast.AsyncFunctionDef),
            ):
                self.visit(child)
            else:
                self._visit_nested_definitions(child)


def analyze_code_paths(
    source: str,
    file: str = "<memory>",
) -> list[CodePath]:
    """
    Analyze approximate execution paths for functions and methods.

    Returns an empty list when the source contains a syntax error.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    collector = _FunctionCollector(file)

    for node in tree.body:
        if isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            collector.visit(node)
        else:
            collector._visit_nested_definitions(node)

    return collector.results


def find_code_paths_by_name(
    results: list[CodePath],
    name: str,
) -> list[CodePath]:
    return [item for item in results if item.name == name]


def find_code_paths_by_kind(
    results: list[CodePath],
    kind: str,
) -> list[CodePath]:
    return [item for item in results if item.kind == kind]


def find_code_paths_for_file(
    results: list[CodePath],
    file: str,
) -> list[CodePath]:
    return [item for item in results if item.file == file]


def build_code_path_report(
    results: list[CodePath],
) -> str:
    lines = [
        "FLY-CODER CODE PATH REPORT",
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
                f"  Lines: {item.start_line}-{item.end_line}",
                f"  Estimated paths: {item.path_count}",
                f"  Exit points: {item.exit_points}",
                f"  Branches: {item.branches}",
                f"  Loops: {item.loops}",
                f"  Exceptions: {item.exceptions}",
                f"  Returns: {item.returns}",
                f"  Raises: {item.raises}",
                f"  Max nesting: {item.max_nesting}",
                "",
            ]
        )

    return "\n".join(lines).rstrip()