from pathlib import Path

from flycoder.tools.filesystem import Workspace
from flycoder.tools.impact import (
    ImpactAnalysis,
    ImpactItem,
    analyze_impact,
    build_impact_report,
    find_impact_by_file,
    find_impact_by_kind,
    find_impact_by_symbol,
)
from flycoder.tools.symbols import Relationship, Symbol, SymbolGraph


def make_workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "auth.py").write_text(
        """
def login(username, password):
    return validate(username, password)

def validate(username, password):
    return bool(username and password)
""",
        encoding="utf-8",
    )

    (tmp_path / "users.py").write_text(
        """
def get_user(user_id):
    return user_id
""",
        encoding="utf-8",
    )

    (tmp_path / "config.py").write_text(
        """
DEBUG = True
""",
        encoding="utf-8",
    )

    return Workspace(tmp_path)


def make_symbols(workspace: Workspace) -> list[Symbol]:
    auth = workspace.root / "auth.py"

    return [
        Symbol(
            name="login",
            kind="function",
            file=auth.as_posix(),
            line=2,
            end_line=3,
        ),
        Symbol(
            name="validate",
            kind="function",
            file=auth.as_posix(),
            line=5,
            end_line=6,
        ),
    ]


def test_empty_task_returns_empty_analysis(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact("", workspace)

    assert isinstance(analysis, ImpactAnalysis)
    assert analysis.task == ""
    assert analysis.items == []


def test_task_is_normalized(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "  Fix   authentication   login  ",
        workspace,
    )

    assert analysis.task == "Fix authentication login"


def test_file_name_can_be_identified(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    files = find_impact_by_kind(analysis, "file")

    assert any(item.path == "auth.py" for item in files)


def test_symbol_name_can_be_identified(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    matches = find_impact_by_symbol(analysis, "login")

    assert len(matches) == 1
    assert matches[0].kind == "symbol"
    assert matches[0].name == "login"


def test_symbol_match_has_higher_score_than_file_match(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    login = find_impact_by_symbol(analysis, "login")[0]
    auth_files = [
        item
        for item in find_impact_by_kind(analysis, "file")
        if item.path == "auth.py"
    ]

    assert auth_files
    assert login.score > auth_files[0].score


def test_multiple_matching_symbols_are_preserved(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login validation",
        workspace,
        symbols=symbols,
    )

    names = {
        item.name
        for item in find_impact_by_kind(analysis, "symbol")
    }

    assert {"login", "validate"} <= names


def test_related_symbol_is_added_from_graph(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    graph = SymbolGraph()
    graph.add_relationships(
        [
            # login -> validate
            Relationship(
                source="login",
                relation="calls",
                target="validate",
                file="auth.py",
                line=3,
            )
        ]
    )

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
        symbol_graph=graph,
    )

    related = find_impact_by_kind(analysis, "related_symbol")

    assert any(item.name == "validate" for item in related)


def test_no_graph_means_no_related_symbols(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    assert find_impact_by_kind(analysis, "related_symbol") == []


def test_find_impact_by_file(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    matches = find_impact_by_file(analysis, "auth.py")

    assert matches
    assert all(item.path == "auth.py" for item in matches)


def test_find_impact_by_file_normalizes_separators(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = [
        Symbol(
            name="login",
            kind="function",
            file="auth.py",
            line=1,
            end_line=2,
        )
    ]

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    assert find_impact_by_file(analysis, "auth.py")


def test_find_impact_by_kind_is_case_insensitive(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    assert find_impact_by_kind(analysis, "FILE")


def test_find_impact_by_symbol_is_case_insensitive(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    assert find_impact_by_symbol(analysis, "LOGIN")


def test_impact_items_are_sorted_by_score(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
    )

    scores = [item.score for item in analysis.items]

    assert scores == sorted(scores, reverse=True)


def test_related_items_have_reason(tmp_path):
    workspace = make_workspace(tmp_path)
    symbols = make_symbols(workspace)

    graph = SymbolGraph()
    graph.add_relationships(
        [
            Relationship(
                source="login",
                relation="calls",
                target="validate",
                file="auth.py",
                line=3,
            )
        ]
    )

    analysis = analyze_impact(
        "Fix login",
        workspace,
        symbols=symbols,
        symbol_graph=graph,
    )

    related = find_impact_by_kind(analysis, "related_symbol")

    assert related
    assert related[0].reason


def test_report_contains_task(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    report = build_impact_report(analysis)

    assert "Fix authentication" in report


def test_report_contains_affected_file(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    report = build_impact_report(analysis)

    assert "auth.py" in report


def test_report_contains_reason(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    report = build_impact_report(analysis)

    assert "Reason:" in report


def test_report_handles_empty_analysis():
    analysis = ImpactAnalysis(task="", items=[])

    report = build_impact_report(analysis)

    assert "No affected areas identified." in report


def test_report_numbers_items(tmp_path):
    workspace = make_workspace(tmp_path)

    analysis = analyze_impact(
        "Fix authentication",
        workspace,
    )

    report = build_impact_report(analysis)

    assert "1." in report


def test_impact_item_dataclass():
    item = ImpactItem(
        path="auth.py",
        kind="file",
        reason="matched authentication",
        score=3,
    )

    assert item.path == "auth.py"
    assert item.kind == "file"
    assert item.score == 3


def test_analysis_files_returns_unique_paths():
    analysis = ImpactAnalysis(
        task="Fix login",
        items=[
            ImpactItem(path="auth.py", kind="file"),
            ImpactItem(path="auth.py", kind="symbol", name="login"),
            ImpactItem(path="users.py", kind="file"),
        ],
    )

    assert analysis.files == ["auth.py", "users.py"]


def test_analysis_symbols_returns_symbol_items():
    analysis = ImpactAnalysis(
        task="Fix login",
        items=[
            ImpactItem(path="auth.py", kind="file"),
            ImpactItem(path="auth.py", kind="symbol", name="login"),
            ImpactItem(path="auth.py", kind="related_symbol", name="validate"),
        ],
    )

    symbols = analysis.symbols

    assert len(symbols) == 1
    assert symbols[0].name == "login"
