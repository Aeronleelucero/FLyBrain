"""Change and impact analysis tools for FLY-CODER."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re

from flycoder.tools.filesystem import Workspace
from flycoder.tools.symbols import Symbol, SymbolGraph


@dataclass
class ImpactItem:
    """One potentially affected project item."""

    path: str
    kind: str
    name: str | None = None
    reason: str = ""
    score: int = 0


@dataclass
class ImpactAnalysis:
    """Structured result of change and impact analysis."""

    task: str
    items: list[ImpactItem] = field(default_factory=list)

    @property
    def files(self) -> list[str]:
        """Return affected file paths in stable order."""
        return list(dict.fromkeys(item.path for item in self.items))

    @property
    def symbols(self) -> list[ImpactItem]:
        """Return directly matched symbol items."""
        return [
            item
            for item in self.items
            if item.kind == "symbol"
        ]


# Small deterministic vocabulary for common coding terminology.
# This deliberately stays narrow; broader semantic matching belongs
# to later reasoning phases.
_TERM_ALIASES = {
    "authentication": {"authentication", "authenticate", "auth"},
    "authorization": {"authorization", "authorize", "auth"},
    "validation": {"validation", "validate", "valid"},
    "configuration": {"configuration", "configure", "config"},
    "database": {"database", "db"},
    "connection": {"connection", "connect"},
    "documentation": {"documentation", "document", "docs"},
    "registration": {"registration", "register", "signup"},
    "logging": {"logging", "log"},
}


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic matching."""
    return " ".join(value.strip().lower().split())


def _expand_term(term: str) -> set[str]:
    """Return deterministic aliases for a task term."""
    normalized = _normalize_text(term)

    aliases = _TERM_ALIASES.get(normalized)

    if aliases:
        return aliases

    return {normalized}


def _task_terms(task: str) -> list[str]:
    """Extract useful terms from a task description."""
    normalized = _normalize_text(task)

    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "so",
        "that",
        "the",
        "this",
        "to",
        "use",
        "with",
    }

    words = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*",
        normalized,
    )

    return [
        word
        for word in words
        if len(word) >= 3 and word not in stop_words
    ]


def _name_parts(name: str) -> set[str]:
    """Split a file or symbol name into comparable parts."""
    normalized = _normalize_text(name)

    parts = re.findall(
        r"[a-zA-Z_][a-zA-Z0-9_]*",
        normalized,
    )

    expanded: set[str] = set()

    for part in parts:
        expanded.add(part)

        # Also support snake_case / kebab-like names.
        for subpart in re.split(r"[_\-]", part):
            if subpart:
                expanded.add(subpart)

    return expanded


def _term_matches_name(term: str, name: str) -> bool:
    """Return whether a task term matches a symbol or file name."""
    name_parts = _name_parts(name)

    for alias in _expand_term(term):
        if alias in name_parts:
            return True

    return False


def _score_symbol(
    symbol: Symbol,
    terms: list[str],
) -> tuple[int, str]:
    """Score a symbol against task terms."""
    score = 0
    reasons: list[str] = []

    for term in terms:
        if _term_matches_name(term, symbol.name):
            score += 5
            reasons.append(
                f"task term matches symbol '{symbol.name}'"
            )

    path_name = Path(symbol.file).name

    for term in terms:
        if _term_matches_name(term, path_name):
            score += 2
            reasons.append(
                f"task term matches file '{path_name}'"
            )

    return score, "; ".join(dict.fromkeys(reasons))


def _workspace_relative_path(
    workspace: Workspace,
    path: str,
) -> str:
    """Convert an analysis path to a workspace-relative POSIX path."""
    candidate = Path(path)

    if not candidate.is_absolute():
        return candidate.as_posix()

    try:
        return candidate.relative_to(workspace.root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _score_file(
    path: str,
    terms: list[str],
) -> tuple[int, str]:
    """Score a project file against task terms."""
    filename = Path(path).name
    stem = Path(path).stem

    score = 0
    reasons: list[str] = []

    for term in terms:
        if _term_matches_name(term, filename):
            score += 3
            reasons.append(
                f"task term matches file '{filename}'"
            )

        if _term_matches_name(term, stem):
            score += 2
            reasons.append(
                f"task term matches module '{stem}'"
            )

    return score, "; ".join(dict.fromkeys(reasons))


def _related_symbol_names(
    symbol: Symbol,
    symbol_graph: SymbolGraph | None,
) -> set[str]:
    """Return symbol names related to a symbol through the graph."""
    if symbol_graph is None:
        return set()

    related: set[str] = set()

    for relationship in symbol_graph.get_relationships():
        if relationship.source == symbol.name:
            related.add(relationship.target)

        if relationship.target == symbol.name:
            related.add(relationship.source)

    return related


def _add_file_item(
    items: list[ImpactItem],
    path: str,
    reason: str,
    score: int,
) -> None:
    """Add a file impact item unless the exact item already exists."""
    normalized_path = path.replace("\\", "/")

    for item in items:
        if (
            item.kind == "file"
            and item.path.replace("\\", "/") == normalized_path
        ):
            if score > item.score:
                item.score = score

            if reason and reason not in item.reason:
                item.reason = (
                    f"{item.reason}; {reason}"
                    if item.reason
                    else reason
                )

            return

    items.append(
        ImpactItem(
            path=path,
            kind="file",
            reason=reason,
            score=score,
        )
    )


def analyze_impact(
    task: str,
    workspace: Workspace,
    symbols: list[Symbol] | None = None,
    symbol_graph: SymbolGraph | None = None,
) -> ImpactAnalysis:
    """
    Analyze project areas that may be affected by a coding task.

    This stage is intentionally conservative and analysis-only. It uses
    task terms, project file names, symbols, and known symbol relationships
    to identify potentially relevant areas. It does not modify source code.
    """
    normalized_task = " ".join(task.strip().split())

    if not normalized_task:
        return ImpactAnalysis(
            task="",
            items=[],
        )

    terms = _task_terms(normalized_task)
    items: list[ImpactItem] = []

    project_files = workspace.list_files()

    for path in project_files:
        score, reason = _score_file(path, terms)

        if score > 0:
            _add_file_item(
                items,
                path=path,
                reason=reason,
                score=score,
            )

    for symbol in symbols or []:
        score, reason = _score_symbol(
            symbol,
            terms,
        )

        if score <= 0:
            continue

        # A matched symbol necessarily makes its containing file
        # potentially affected, even when the task term matched only
        # the symbol name.
        relative_symbol_file = _workspace_relative_path(
            workspace,
            symbol.file,
        )

        _add_file_item(
            items,
            path=relative_symbol_file,
            reason=(
                f"contains matched symbol '{symbol.name}'"
            ),
            score=max(score - 1, 1),
        )

        items.append(
            ImpactItem(
                path=relative_symbol_file,
                kind="symbol",
                name=symbol.name,
                reason=reason,
                score=score,
            )
        )

    # Expand directly related symbols from the existing symbol graph.
    if symbol_graph is not None and symbols:
        matched_names = {
            item.name
            for item in items
            if item.kind == "symbol" and item.name
        }

        symbol_by_name = {
            symbol.name: symbol
            for symbol in symbols
        }

        for matched_name in matched_names:
            matched_symbol = symbol_by_name.get(
                matched_name
            )

            if matched_symbol is None:
                continue

            for related_name in _related_symbol_names(
                matched_symbol,
                symbol_graph,
            ):
                related_symbol = symbol_by_name.get(
                    related_name
                )

                if related_symbol is None:
                    continue

                relative_related_file = _workspace_relative_path(
                    workspace,
                    related_symbol.file,
                )

                items.append(
                    ImpactItem(
                        path=relative_related_file,
                        kind="related_symbol",
                        name=related_symbol.name,
                        reason=(
                            f"related to matched symbol "
                            f"'{matched_symbol.name}'"
                        ),
                        score=1,
                    )
                )

                _add_file_item(
                    items,
                    path=relative_related_file,
                    reason=(
                        f"contains related symbol "
                        f"'{related_symbol.name}'"
                    ),
                    score=1,
                )

    # Higher-impact items appear first while preserving stable ordering
    # for items with equal scores.
    items.sort(
        key=lambda item: (
            -item.score,
            item.path,
            item.kind,
            item.name or "",
        )
    )

    return ImpactAnalysis(
        task=normalized_task,
        items=items,
    )


def find_impact_by_kind(
    analysis: ImpactAnalysis,
    kind: str,
) -> list[ImpactItem]:
    """Return impact items matching a kind."""
    normalized_kind = kind.strip().lower()

    return [
        item
        for item in analysis.items
        if item.kind.lower() == normalized_kind
    ]


def find_impact_by_file(
    analysis: ImpactAnalysis,
    path: str,
) -> list[ImpactItem]:
    """Return impact items associated with a file."""
    normalized_path = path.replace("\\", "/")

    return [
        item
        for item in analysis.items
        if item.path.replace("\\", "/") == normalized_path
    ]


def find_impact_by_symbol(
    analysis: ImpactAnalysis,
    name: str,
) -> list[ImpactItem]:
    """Return impact items associated with a symbol name."""
    normalized_name = name.strip().lower()

    return [
        item
        for item in analysis.items
        if item.name and item.name.lower() == normalized_name
    ]


def build_impact_report(
    analysis: ImpactAnalysis,
) -> str:
    """Build a readable impact analysis report."""
    lines = [
        "FLY-CODER IMPACT ANALYSIS",
        "=" * 27,
        "",
        f"Task: {analysis.task}",
        "",
        "Potentially affected areas:",
    ]

    if not analysis.items:
        lines.append(
            "  No affected areas identified."
        )
        return "\n".join(lines)

    for index, item in enumerate(
        analysis.items,
        start=1,
    ):
        name = (
            f"::{item.name}"
            if item.name
            else ""
        )

        lines.append(
            f"  {index}. [{item.kind}] "
            f"{item.path}{name} "
            f"(score={item.score})"
        )

        if item.reason:
            lines.append(
                f"     Reason: {item.reason}"
            )

    return "\n".join(lines)
