"""Symbol analysis tools for FLY-CODER."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass
class Symbol:
    """Represent a Python symbol discovered by the analyzer."""

    name: str
    kind: str
    file: str
    line: int
    end_line: int
    parent: str | None = None


@dataclass
class ImportInfo:
    """Represent an import discovered by the analyzer."""

    module: str
    name: str | None
    alias: str | None
    file: str
    line: int


@dataclass
class CallInfo:
    """Represent a function or method call discovered by the analyzer."""

    name: str
    file: str
    line: int
    parent: str | None = None


@dataclass
class InheritanceInfo:
    """Represent a class inheritance relationship."""

    class_name: str
    base_name: str
    file: str
    line: int


@dataclass
class DecoratorInfo:
    """Represent a decorator applied to a symbol."""

    symbol_name: str
    decorator_name: str
    file: str
    line: int


@dataclass
class Relationship:
    """Represent a relationship between two code entities."""

    source: str
    relation: str
    target: str
    file: str
    line: int


def analyze_symbols(
    content: str,
    file_path: str = "<memory>",
) -> list[Symbol]:
    """Analyze Python source and return discovered symbols."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    symbols: list[Symbol] = []

    def visit_node(
        node: ast.AST,
        parent: str | None = None,
    ) -> None:
        """Recursively collect classes, functions, and methods."""

        if isinstance(node, ast.ClassDef):
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="class",
                    file=file_path,
                    line=node.lineno,
                    end_line=getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),
                    parent=parent,
                )
            )

            for child in node.body:
                visit_node(
                    child,
                    parent=node.name,
                )

            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = (
                "method"
                if parent is not None
                else "function"
            )

            symbols.append(
                Symbol(
                    name=node.name,
                    kind=kind,
                    file=file_path,
                    line=node.lineno,
                    end_line=getattr(
                        node,
                        "end_lineno",
                        node.lineno,
                    ),
                    parent=parent,
                )
            )

            return

        for child in ast.iter_child_nodes(node):
            visit_node(
                child,
                parent=parent,
            )

    for node in tree.body:
        visit_node(node)

    return symbols


def analyze_imports(
    content: str,
    file_path: str = "<memory>",
) -> list[ImportInfo]:
    """Analyze Python source and return discovered imports."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    imports: list[ImportInfo] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=alias.name,
                        name=None,
                        alias=alias.asname,
                        file=file_path,
                        line=node.lineno,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""

            for alias in node.names:
                imports.append(
                    ImportInfo(
                        module=module,
                        name=alias.name,
                        alias=alias.asname,
                        file=file_path,
                        line=node.lineno,
                    )
                )

    return imports


def analyze_calls(
    content: str,
    file_path: str = "<memory>",
) -> list[CallInfo]:
    """Analyze Python source and return discovered function/method calls."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    calls: list[CallInfo] = []

    def get_call_name(node: ast.AST) -> str | None:
        """Convert a callable AST node into a readable name."""

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)

            return node.attr

        return None

    def visit_node(
        node: ast.AST,
        parent: str | None = None,
    ) -> None:
        """Recursively collect calls and track their containing symbol."""

        if isinstance(node, ast.ClassDef):
            for child in node.body:
                visit_node(
                    child,
                    parent=node.name,
                )

            return

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in node.body:
                visit_node(
                    child,
                    parent=node.name,
                )

            return

        if isinstance(node, ast.Call):
            name = get_call_name(node.func)

            if name is not None:
                calls.append(
                    CallInfo(
                        name=name,
                        file=file_path,
                        line=node.lineno,
                        parent=parent,
                    )
                )

        for child in ast.iter_child_nodes(node):
            visit_node(
                child,
                parent=parent,
            )

    for node in tree.body:
        visit_node(node)

    return calls


def analyze_inheritance(
    content: str,
    file_path: str = "<memory>",
) -> list[InheritanceInfo]:
    """Analyze Python source and return class inheritance relationships."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    inheritance: list[InheritanceInfo] = []

    def get_base_name(node: ast.AST) -> str | None:
        """Convert a base-class AST node into a readable name."""

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)

            return node.attr

        return None

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        for base in node.bases:
            base_name = get_base_name(base)

            if base_name is not None:
                inheritance.append(
                    InheritanceInfo(
                        class_name=node.name,
                        base_name=base_name,
                        file=file_path,
                        line=node.lineno,
                    )
                )

    return inheritance


def analyze_decorators(
    content: str,
    file_path: str = "<memory>",
) -> list[DecoratorInfo]:
    """Analyze Python source and return decorators applied to symbols."""

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    decorators: list[DecoratorInfo] = []

    def get_decorator_name(node: ast.AST) -> str | None:
        """Convert a decorator AST node into a readable name."""

        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parts: list[str] = []
            current: ast.AST | None = node

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)
                parts.reverse()
                return ".".join(parts)

            return node.attr

        if isinstance(node, ast.Call):
            return get_decorator_name(node.func)

        return None

    for node in ast.walk(tree):
        if not isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):
            continue

        for decorator in node.decorator_list:
            decorator_name = get_decorator_name(decorator)

            if decorator_name is not None:
                decorators.append(
                    DecoratorInfo(
                        symbol_name=node.name,
                        decorator_name=decorator_name,
                        file=file_path,
                        line=node.lineno,
                    )
                )

    return decorators


def analyze_relationships(
    content: str,
    file_path: str = "<memory>",
) -> list[Relationship]:
    """Analyze and combine relationships between code entities."""

    relationships: list[Relationship] = []

    imports = analyze_imports(
        content,
        file_path,
    )

    calls = analyze_calls(
        content,
        file_path,
    )

    inheritance = analyze_inheritance(
        content,
        file_path,
    )

    decorators = analyze_decorators(
        content,
        file_path,
    )

    for item in imports:
        target = item.name or item.module

        relationships.append(
            Relationship(
                source=file_path,
                relation="imports",
                target=target,
                file=item.file,
                line=item.line,
            )
        )

    for item in calls:
        source = item.parent or item.file

        relationships.append(
            Relationship(
                source=source,
                relation="calls",
                target=item.name,
                file=item.file,
                line=item.line,
            )
        )

    for item in inheritance:
        relationships.append(
            Relationship(
                source=item.class_name,
                relation="inherits",
                target=item.base_name,
                file=item.file,
                line=item.line,
            )
        )

    for item in decorators:
        relationships.append(
            Relationship(
                source=item.symbol_name,
                relation="decorated_by",
                target=item.decorator_name,
                file=item.file,
                line=item.line,
            )
        )

    return relationships


class SymbolGraph:
    """Represent and query relationships between code entities."""

    def __init__(
        self,
        relationships: list[Relationship] | None = None,
    ) -> None:
        """Initialize the graph with optional relationships."""

        self.relationships = relationships or []

    def add_relationship(
        self,
        relationship: Relationship,
    ) -> None:
        """Add a relationship to the graph."""

        self.relationships.append(relationship)

    def add_relationships(
        self,
        relationships: list[Relationship],
    ) -> None:
        """Add multiple relationships to the graph."""

        self.relationships.extend(relationships)

    def get_relationships(
        self,
        source: str | None = None,
        relation: str | None = None,
        target: str | None = None,
    ) -> list[Relationship]:
        """Return relationships matching the supplied filters."""

        results: list[Relationship] = []

        for relationship in self.relationships:
            if source is not None and relationship.source != source:
                continue

            if relation is not None and relationship.relation != relation:
                continue

            if target is not None and relationship.target != target:
                continue

            results.append(relationship)

        return results

    def get_related(
        self,
        symbol: str,
    ) -> list[Relationship]:
        """Return all relationships involving a symbol."""

        return [
            relationship
            for relationship in self.relationships
            if (
                relationship.source == symbol
                or relationship.target == symbol
            )
        ]

    def get_callers(
        self,
        symbol: str,
    ) -> list[str]:
        """Return symbols that call the specified symbol."""

        return [
            relationship.source
            for relationship in self.get_relationships(
                relation="calls",
                target=symbol,
            )
        ]

    def get_callees(
        self,
        symbol: str,
    ) -> list[str]:
        """Return symbols called by the specified symbol."""

        return [
            relationship.target
            for relationship in self.get_relationships(
                source=symbol,
                relation="calls",
            )
        ]

    def get_inheritors(
        self,
        symbol: str,
    ) -> list[str]:
        """Return classes that inherit from the specified class."""

        return [
            relationship.source
            for relationship in self.get_relationships(
                relation="inherits",
                target=symbol,
            )
        ]

    def get_importers(
        self,
        symbol: str,
    ) -> list[str]:
        """Return files that import the specified symbol."""

        return [
            relationship.source
            for relationship in self.get_relationships(
                relation="imports",
                target=symbol,
            )
        ]

    def get_decorators(
        self,
        symbol: str,
    ) -> list[str]:
        """Return decorators applied to the specified symbol."""

        return [
            relationship.target
            for relationship in self.get_relationships(
                source=symbol,
                relation="decorated_by",
            )
        ]


def find_symbols_by_kind(
    symbols: list[Symbol],
    kind: str,
) -> list[Symbol]:
    """Return symbols matching a specific kind."""

    return [
        symbol
        for symbol in symbols
        if symbol.kind == kind
    ]
def build_symbol_report(
    symbols: list[Symbol],
    imports: list[ImportInfo] | None = None,
    calls: list[CallInfo] | None = None,
    inheritance: list[InheritanceInfo] | None = None,
    decorators: list[DecoratorInfo] | None = None,
    relationships: list[Relationship] | None = None,
) -> str:
    """Build a readable report from discovered symbol intelligence."""

    imports = imports or []
    calls = calls or []
    inheritance = inheritance or []
    decorators = decorators or []
    relationships = relationships or []

    lines: list[str] = []

    files = sorted(
        {
            symbol.file
            for symbol in symbols
        }
        | {
            item.file
            for item in imports
        }
        | {
            item.file
            for item in calls
        }
        | {
            item.file
            for item in inheritance
        }
        | {
            item.file
            for item in decorators
        }
    )

    if not files:
        files = ["<unknown>"]

    for file_path in files:
        lines.append("=" * 60)
        lines.append("FLY-CODER SYMBOL REPORT")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"File: {file_path}")
        lines.append("")

        file_symbols = [
            symbol
            for symbol in symbols
            if symbol.file == file_path
        ]

        file_imports = [
            item
            for item in imports
            if item.file == file_path
        ]

        file_calls = [
            item
            for item in calls
            if item.file == file_path
        ]

        file_inheritance = [
            item
            for item in inheritance
            if item.file == file_path
        ]

        file_decorators = [
            item
            for item in decorators
            if item.file == file_path
        ]

        file_relationships = [
            item
            for item in relationships
            if item.file == file_path
        ]

        lines.append(
            f"Symbols: {len(file_symbols)}"
        )

        if file_symbols:
            lines.append("")

            for symbol in file_symbols:
                location = (
                    f"{symbol.line}-{symbol.end_line}"
                )

                if symbol.parent:
                    lines.append(
                        f"  {symbol.kind:<10}"
                        f"{symbol.name} "
                        f"[{location}] "
                        f"parent={symbol.parent}"
                    )
                else:
                    lines.append(
                        f"  {symbol.kind:<10}"
                        f"{symbol.name} "
                        f"[{location}]"
                    )
        else:
            lines.append(
                "  No symbols detected."
            )

        lines.append("")
        lines.append(
            f"Imports: {len(file_imports)}"
        )

        if file_imports:
            for item in file_imports:
                imported_name = (
                    item.name
                    if item.name is not None
                    else item.module
                )

                alias = (
                    f" as {item.alias}"
                    if item.alias
                    else ""
                )

                lines.append(
                    f"  {imported_name}{alias} "
                    f"[line {item.line}]"
                )
        else:
            lines.append(
                "  No imports detected."
            )

        lines.append("")
        lines.append(
            f"Calls: {len(file_calls)}"
        )

        if file_calls:
            for item in file_calls:
                parent = (
                    f" in {item.parent}"
                    if item.parent
                    else ""
                )

                lines.append(
                    f"  {item.name}{parent} "
                    f"[line {item.line}]"
                )
        else:
            lines.append(
                "  No calls detected."
            )

        lines.append("")
        lines.append(
            f"Inheritance: {len(file_inheritance)}"
        )

        if file_inheritance:
            for item in file_inheritance:
                lines.append(
                    f"  {item.class_name} "
                    f"-> {item.base_name} "
                    f"[line {item.line}]"
                )
        else:
            lines.append(
                "  No inheritance detected."
            )

        lines.append("")
        lines.append(
            f"Decorators: {len(file_decorators)}"
        )

        if file_decorators:
            for item in file_decorators:
                lines.append(
                    f"  {item.symbol_name} "
                    f"-> @{item.decorator_name} "
                    f"[line {item.line}]"
                )
        else:
            lines.append(
                "  No decorators detected."
            )

        lines.append("")
        lines.append(
            f"Relationships: {len(file_relationships)}"
        )

        if file_relationships:
            for item in file_relationships:
                lines.append(
                    f"  {item.source} "
                    f"--{item.relation}--> "
                    f"{item.target} "
                    f"[line {item.line}]"
                )
        else:
            lines.append(
                "  No relationships detected."
            )

        lines.append("")
        lines.append("=" * 60)
        lines.append("")

    return "\n".join(lines).rstrip()
