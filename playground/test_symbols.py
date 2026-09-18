import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from flycoder.state import CodingState
from flycoder.tools.symbols import (
    CallInfo,
    DecoratorInfo,
    ImportInfo,
    InheritanceInfo,
    Relationship,
    Symbol,
    SymbolGraph,
    analyze_calls,
    analyze_decorators,
    analyze_imports,
    analyze_inheritance,
    analyze_relationships,
    analyze_symbols,
    build_symbol_report,
    find_symbols_by_kind,
)


def test_analyze_symbols_finds_class():
    source = """
class Example:
    pass
"""

    symbols = analyze_symbols(
        source,
        "example.py",
    )

    assert symbols == [
        Symbol(
            name="Example",
            kind="class",
            file="example.py",
            line=2,
            end_line=3,
            parent=None,
        )
    ]


def test_analyze_symbols_finds_method():
    source = """
class Example:

    def hello(self):
        pass
"""

    symbols = analyze_symbols(
        source,
        "example.py",
    )

    assert len(symbols) == 2

    method = symbols[1]

    assert method.name == "hello"
    assert method.kind == "method"
    assert method.parent == "Example"
    assert method.line == 4
    assert method.end_line == 5


def test_analyze_symbols_finds_function():
    source = """
def standalone():
    pass
"""

    symbols = analyze_symbols(
        source,
        "example.py",
    )

    assert len(symbols) == 1

    function = symbols[0]

    assert function.name == "standalone"
    assert function.kind == "function"
    assert function.parent is None
    assert function.line == 2
    assert function.end_line == 3


def test_find_symbols_by_kind():
    source = """
class Example:

    def hello(self):
        pass


def standalone():
    pass
"""

    symbols = analyze_symbols(
        source,
        "example.py",
    )

    classes = find_symbols_by_kind(
        symbols,
        "class",
    )

    methods = find_symbols_by_kind(
        symbols,
        "method",
    )

    functions = find_symbols_by_kind(
        symbols,
        "function",
    )

    assert len(classes) == 1
    assert classes[0].name == "Example"

    assert len(methods) == 1
    assert methods[0].name == "hello"

    assert len(functions) == 1
    assert functions[0].name == "standalone"


def test_analyze_symbols_handles_invalid_python():
    source = """
class Broken(
"""

    symbols = analyze_symbols(
        source,
        "broken.py",
    )

    assert symbols == []


def test_analyze_imports_finds_regular_import():
    source = """
import os
import argparse as ap
"""

    imports = analyze_imports(
        source,
        "example.py",
    )

    assert imports == [
        ImportInfo(
            module="os",
            name=None,
            alias=None,
            file="example.py",
            line=2,
        ),
        ImportInfo(
            module="argparse",
            name=None,
            alias="ap",
            file="example.py",
            line=3,
        ),
    ]


def test_analyze_imports_finds_from_import():
    source = """
from pathlib import Path
from flycoder.state import CodingState as State
"""

    imports = analyze_imports(
        source,
        "example.py",
    )

    assert imports == [
        ImportInfo(
            module="pathlib",
            name="Path",
            alias=None,
            file="example.py",
            line=2,
        ),
        ImportInfo(
            module="flycoder.state",
            name="CodingState",
            alias="State",
            file="example.py",
            line=3,
        ),
    ]


def test_analyze_imports_handles_multiple_names():
    source = """
from os import path, getcwd as cwd
"""

    imports = analyze_imports(
        source,
        "example.py",
    )

    assert imports == [
        ImportInfo(
            module="os",
            name="path",
            alias=None,
            file="example.py",
            line=2,
        ),
        ImportInfo(
            module="os",
            name="getcwd",
            alias="cwd",
            file="example.py",
            line=2,
        ),
    ]


def test_analyze_imports_handles_invalid_python():
    source = """
from broken import
"""

    imports = analyze_imports(
        source,
        "broken.py",
    )

    assert imports == []


def test_analyze_calls_finds_function_call():
    source = """
def process():
    result = calculate_total(items)
    return result
"""

    calls = analyze_calls(
        source,
        "example.py",
    )

    assert calls == [
        CallInfo(
            name="calculate_total",
            file="example.py",
            line=3,
            parent="process",
        )
    ]


def test_analyze_calls_finds_method_call():
    source = """
def process(user):
    user.save()
"""

    calls = analyze_calls(
        source,
        "example.py",
    )

    assert calls == [
        CallInfo(
            name="user.save",
            file="example.py",
            line=3,
            parent="process",
        )
    ]


def test_analyze_calls_finds_calls_inside_method():
    source = """
class Service:

    def process(self, data):
        result = helper(data)
        self.save(result)
        return result
"""

    calls = analyze_calls(
        source,
        "example.py",
    )

    assert calls == [
        CallInfo(
            name="helper",
            file="example.py",
            line=5,
            parent="process",
        ),
        CallInfo(
            name="self.save",
            file="example.py",
            line=6,
            parent="process",
        ),
    ]


def test_analyze_calls_finds_multiple_calls():
    source = """
print("hello")
calculate_total(items)
service.process(data)
"""

    calls = analyze_calls(
        source,
        "example.py",
    )

    assert calls == [
        CallInfo(
            name="print",
            file="example.py",
            line=2,
            parent=None,
        ),
        CallInfo(
            name="calculate_total",
            file="example.py",
            line=3,
            parent=None,
        ),
        CallInfo(
            name="service.process",
            file="example.py",
            line=4,
            parent=None,
        ),
    ]


def test_analyze_calls_handles_invalid_python():
    source = """
def broken(
"""

    calls = analyze_calls(
        source,
        "broken.py",
    )

    assert calls == []


def test_analyze_calls_handles_nested_attribute():
    source = """
def process():
    service.api.send(data)
"""

    calls = analyze_calls(
        source,
        "example.py",
    )

    assert calls == [
        CallInfo(
            name="service.api.send",
            file="example.py",
            line=3,
            parent="process",
        )
    ]


def test_analyze_inheritance_finds_single_base():
    source = """
class Parent:
    pass


class Child(Parent):
    pass
"""

    inheritance = analyze_inheritance(
        source,
        "example.py",
    )

    assert inheritance == [
        InheritanceInfo(
            class_name="Child",
            base_name="Parent",
            file="example.py",
            line=6,
        )
    ]


def test_analyze_inheritance_finds_multiple_bases():
    source = """
class ParentA:
    pass


class ParentB:
    pass


class Child(ParentA, ParentB):
    pass
"""

    inheritance = analyze_inheritance(
        source,
        "example.py",
    )

    assert inheritance == [
        InheritanceInfo(
            class_name="Child",
            base_name="ParentA",
            file="example.py",
            line=10,
        ),
        InheritanceInfo(
            class_name="Child",
            base_name="ParentB",
            file="example.py",
            line=10,
        ),
    ]


def test_analyze_inheritance_handles_attribute_base():
    source = """
class Child(package.Parent):
    pass
"""

    inheritance = analyze_inheritance(
        source,
        "example.py",
    )

    assert inheritance == [
        InheritanceInfo(
            class_name="Child",
            base_name="package.Parent",
            file="example.py",
            line=2,
        )
    ]


def test_analyze_inheritance_handles_invalid_python():
    source = """
class Broken(
"""

    inheritance = analyze_inheritance(
        source,
        "broken.py",
    )

    assert inheritance == []


def test_analyze_decorators_finds_simple_decorator():
    source = """
@login_required
def users():
    pass
"""

    decorators = analyze_decorators(
        source,
        "example.py",
    )

    assert decorators == [
        DecoratorInfo(
            symbol_name="users",
            decorator_name="login_required",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_decorators_finds_attribute_decorator():
    source = """
@app.route("/users")
def users():
    pass
"""

    decorators = analyze_decorators(
        source,
        "example.py",
    )

    assert decorators == [
        DecoratorInfo(
            symbol_name="users",
            decorator_name="app.route",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_decorators_finds_multiple_decorators():
    source = """
@app.route("/users")
@login_required
def users():
    pass
"""

    decorators = analyze_decorators(
        source,
        "example.py",
    )

    assert decorators == [
        DecoratorInfo(
            symbol_name="users",
            decorator_name="app.route",
            file="example.py",
            line=4,
        ),
        DecoratorInfo(
            symbol_name="users",
            decorator_name="login_required",
            file="example.py",
            line=4,
        ),
    ]


def test_analyze_decorators_handles_called_decorator():
    source = """
@route("/users")
def users():
    pass
"""

    decorators = analyze_decorators(
        source,
        "example.py",
    )

    assert decorators == [
        DecoratorInfo(
            symbol_name="users",
            decorator_name="route",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_decorators_finds_class_decorator():
    source = """
@dataclass
class User:
    pass
"""

    decorators = analyze_decorators(
        source,
        "example.py",
    )

    assert decorators == [
        DecoratorInfo(
            symbol_name="User",
            decorator_name="dataclass",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_decorators_handles_invalid_python():
    source = """
@login_required
def broken(
"""

    decorators = analyze_decorators(
        source,
        "broken.py",
    )

    assert decorators == []


def test_analyze_relationships_combines_inheritance():
    source = """
class Parent:
    pass


class Child(Parent):
    pass
"""

    relationships = analyze_relationships(
        source,
        "example.py",
    )

    assert relationships == [
        Relationship(
            source="Child",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=6,
        )
    ]


def test_analyze_relationships_combines_decorator():
    source = """
@login_required
def users():
    pass
"""

    relationships = analyze_relationships(
        source,
        "example.py",
    )

    assert relationships == [
        Relationship(
            source="users",
            relation="decorated_by",
            target="login_required",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_relationships_combines_call():
    source = """
def process():
    helper()
"""

    relationships = analyze_relationships(
        source,
        "example.py",
    )

    assert relationships == [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        )
    ]


def test_analyze_relationships_combines_import():
    source = """
from flycoder.state import CodingState
"""

    relationships = analyze_relationships(
        source,
        "example.py",
    )

    assert relationships == [
        Relationship(
            source="example.py",
            relation="imports",
            target="CodingState",
            file="example.py",
            line=2,
        )
    ]


def test_analyze_relationships_combines_multiple_types():
    source = """
from framework import BaseService

@register_service
class UserService(BaseService):

    def process(self):
        helper()
"""

    relationships = analyze_relationships(
        source,
        "example.py",
    )

    assert relationships == [
        Relationship(
            source="example.py",
            relation="imports",
            target="BaseService",
            file="example.py",
            line=2,
        ),
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=8,
        ),
        Relationship(
            source="UserService",
            relation="inherits",
            target="BaseService",
            file="example.py",
            line=5,
        ),
        Relationship(
            source="UserService",
            relation="decorated_by",
            target="register_service",
            file="example.py",
            line=5,
        ),
    ]


def test_analyze_relationships_handles_invalid_python():
    source = """
class Broken(
"""

    relationships = analyze_relationships(
        source,
        "broken.py",
    )

    assert relationships == []


def test_symbol_graph_get_related():
    relationships = [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="Child",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=6,
        ),
    ]

    graph = SymbolGraph(relationships)

    related = graph.get_related("process")

    assert related == [
        relationships[0],
    ]


def test_symbol_graph_get_callers():
    relationships = [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="main",
            relation="calls",
            target="helper",
            file="example.py",
            line=8,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_callers("helper") == [
        "process",
        "main",
    ]


def test_symbol_graph_get_callees():
    relationships = [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="process",
            relation="calls",
            target="save",
            file="example.py",
            line=4,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_callees("process") == [
        "helper",
        "save",
    ]


def test_symbol_graph_get_inheritors():
    relationships = [
        Relationship(
            source="ChildA",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=5,
        ),
        Relationship(
            source="ChildB",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=10,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_inheritors("Parent") == [
        "ChildA",
        "ChildB",
    ]


def test_symbol_graph_get_importers():
    relationships = [
        Relationship(
            source="module_a.py",
            relation="imports",
            target="CodingState",
            file="module_a.py",
            line=2,
        ),
        Relationship(
            source="module_b.py",
            relation="imports",
            target="CodingState",
            file="module_b.py",
            line=4,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_importers("CodingState") == [
        "module_a.py",
        "module_b.py",
    ]


def test_symbol_graph_get_decorators():
    relationships = [
        Relationship(
            source="users",
            relation="decorated_by",
            target="login_required",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="users",
            relation="decorated_by",
            target="route",
            file="example.py",
            line=3,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_decorators("users") == [
        "login_required",
        "route",
    ]


def test_symbol_graph_filters_relationships():
    relationships = [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="process",
            relation="calls",
            target="save",
            file="example.py",
            line=4,
        ),
        Relationship(
            source="Child",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=7,
        ),
    ]

    graph = SymbolGraph(relationships)

    assert graph.get_relationships(
        source="process",
    ) == relationships[:2]

    assert graph.get_relationships(
        relation="calls",
    ) == relationships[:2]

    assert graph.get_relationships(
        target="Parent",
    ) == relationships[2:]

    assert graph.get_relationships(
        source="process",
        relation="calls",
        target="helper",
    ) == relationships[:1]


def test_symbol_graph_add_relationship():
    graph = SymbolGraph()

    relationship = Relationship(
        source="process",
        relation="calls",
        target="helper",
        file="example.py",
        line=3,
    )

    graph.add_relationship(relationship)

    assert graph.relationships == [
        relationship,
    ]


def test_symbol_graph_add_relationships():
    graph = SymbolGraph()

    relationships = [
        Relationship(
            source="process",
            relation="calls",
            target="helper",
            file="example.py",
            line=3,
        ),
        Relationship(
            source="Child",
            relation="inherits",
            target="Parent",
            file="example.py",
            line=6,
        ),
    ]

    graph.add_relationships(relationships)

    assert graph.relationships == relationships


def test_symbol_graph_handles_empty_graph():
    graph = SymbolGraph()

    assert graph.get_related("missing") == []
    assert graph.get_callers("missing") == []
    assert graph.get_callees("missing") == []
    assert graph.get_inheritors("missing") == []
    assert graph.get_importers("missing") == []
    assert graph.get_decorators("missing") == []


def test_coding_state_has_symbol_intelligence_defaults():
    from flycoder.state import CodingState

    state = CodingState(
        task="Analyze the project",
    )

    assert state.symbols == []
    assert state.symbol_imports == []
    assert state.symbol_calls == []
    assert state.symbol_inheritance == []
    assert state.symbol_decorators == []
    assert state.symbol_relationships == []
    assert state.symbol_graph is None


def test_coding_state_accepts_symbol_graph():
    from flycoder.state import CodingState

    relationship = Relationship(
        source="process",
        relation="calls",
        target="helper",
        file="example.py",
        line=3,
    )

    graph = SymbolGraph(
        [relationship],
    )

    state = CodingState(
        task="Analyze the project",
        symbol_relationships=[relationship],
        symbol_graph=graph,
    )

    assert state.symbol_relationships == [
        relationship,
    ]

    assert state.symbol_graph is graph

    assert state.symbol_graph.get_callees(
        "process",
    ) == [
        "helper",
    ]


def test_review_code_populates_symbol_state():
    from flycoder.actions.analysis_actions import review_code_action
    from flycoder.state import CodingState

    state = CodingState(
        task="Review this file",
        current_file="example.py",
        current_file_content=(
            "import os\n\n"
            "class Service:\n"
            "    def run(self):\n"
            "        os.getcwd()\n"
        ),
    )

    result = review_code_action(
        None,
        state,
    )

    assert result.success is True

    assert len(state.symbols) == 2
    assert len(state.symbol_imports) == 1
    assert len(state.symbol_calls) == 1
    assert len(state.symbol_inheritance) == 0
    assert len(state.symbol_decorators) == 0
    assert len(state.symbol_relationships) == 2

    assert state.symbol_graph is not None
    assert state.symbol_graph.get_callees(
        "run",
    ) == [
        "os.getcwd",
    ]


def test_review_code_does_not_duplicate_symbol_state():
    from flycoder.actions.analysis_actions import review_code_action
    from flycoder.state import CodingState

    content = (
        "def helper():\n"
        "    return 42\n"
    )

    state = CodingState(
        task="Review this file",
        current_file="example.py",
        current_file_content=content,
    )

    first = review_code_action(
        None,
        state,
    )

    assert first.success is True

    first_symbol_count = len(
        state.symbols
    )

    first_relationship_count = len(
        state.symbol_relationships
    )

    second = review_code_action(
        None,
        state,
    )

    assert second.success is True

    assert len(state.symbols) == first_symbol_count
    assert (
        len(state.symbol_relationships)
        == first_relationship_count
    )


def test_relevant_files_use_symbol_matching(
    tmp_path,
):
    from flycoder.agent import FlyCoderAgent

    workspace_root = tmp_path

    (
        workspace_root / "agent.py"
    ).write_text(
        "class Agent:\n"
        "    def run(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    (
        workspace_root / "other.py"
    ).write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    agent = FlyCoderAgent(
        workspace_root
    )

    selected = agent.select_relevant_files(
        "Review the Agent",
        [
            "agent.py",
            "other.py",
        ],
        max_files=2,
        workspace=agent.workspace,
    )

    assert selected[0] == "agent.py"


def test_relevant_files_find_symbolgraph(
    tmp_path,
):
    from flycoder.agent import FlyCoderAgent

    workspace_root = tmp_path

    (
        workspace_root / "symbols.py"
    ).write_text(
        "class SymbolGraph:\n"
        "    pass\n",
        encoding="utf-8",
    )

    (
        workspace_root / "other.py"
    ).write_text(
        "class SomethingElse:\n"
        "    pass\n",
        encoding="utf-8",
    )

    agent = FlyCoderAgent(
        workspace_root
    )

    selected = agent.select_relevant_files(
        "Review the SymbolGraph",
        [
            "symbols.py",
            "other.py",
        ],
        max_files=2,
        workspace=agent.workspace,
    )

    assert selected[0] == "symbols.py"


def test_relevant_files_preserve_backward_compatibility():
    from flycoder.agent import FlyCoderAgent

    selected = FlyCoderAgent.select_relevant_files(
        "Review the agent",
        [
            "flycoder/agent.py",
            "flycoder/state.py",
            "experiments/live_trained_agent.py",
        ],
        max_files=2,
    )

    assert selected
    assert selected[0] == "flycoder/agent.py"


def test_build_symbol_report():
    content = (
        "import os\n\n"
        "class Service:\n"
        "    @staticmethod\n"
        "    def run():\n"
        "        os.getcwd()\n"
    )

    symbols = analyze_symbols(
        content,
        "service.py",
    )

    imports = analyze_imports(
        content,
        "service.py",
    )

    calls = analyze_calls(
        content,
        "service.py",
    )

    inheritance = analyze_inheritance(
        content,
        "service.py",
    )

    decorators = analyze_decorators(
        content,
        "service.py",
    )

    relationships = analyze_relationships(
        content,
        "service.py",
    )

    report = build_symbol_report(
        symbols=symbols,
        imports=imports,
        calls=calls,
        inheritance=inheritance,
        decorators=decorators,
        relationships=relationships,
    )

    assert "FLY-CODER SYMBOL REPORT" in report
    assert "File: service.py" in report
    assert "class" in report
    assert "Service" in report
    assert "method" in report
    assert "run" in report
    assert "Imports: 1" in report
    assert "Calls: 1" in report
    assert "Decorators: 1" in report
    assert "os.getcwd" in report
    assert "run --calls--> os.getcwd" in report

def print_symbol_report(
    self,
    state: CodingState,
) -> None:
    """Print the discovered project symbol intelligence."""

    if not (
        state.symbols
        or state.symbol_imports
        or state.symbol_calls
        or state.symbol_inheritance
        or state.symbol_decorators
        or state.symbol_relationships
    ):
        return

    report = build_symbol_report(
        symbols=state.symbols,
        imports=state.symbol_imports,
        calls=state.symbol_calls,
        inheritance=state.symbol_inheritance,
        decorators=state.symbol_decorators,
        relationships=state.symbol_relationships,
    )

    print()
    print(report)
    print()
