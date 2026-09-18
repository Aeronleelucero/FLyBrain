from flycoder.tools.control_flow import (
    ControlFlowNode,
    analyze_control_flow,
    build_control_flow_report,
    find_control_flow_by_kind,
    find_control_flow_for_parent,
)


def test_empty_source():
    nodes = analyze_control_flow("", "empty.py")

    assert nodes == []


def test_simple_if():
    content = """
def check(value):
    if value:
        return True
"""

    nodes = analyze_control_flow(content, "check.py")

    assert len(nodes) == 2

    assert nodes[0].kind == "if"
    assert nodes[0].line == 3
    assert nodes[0].parent == "check"
    assert nodes[0].depth == 0

    assert nodes[1].kind == "return"
    assert nodes[1].line == 4
    assert nodes[1].parent == "check"
    assert nodes[1].depth == 1


def test_if_else():
    content = """
def check(value):
    if value:
        return True
    else:
        return False
"""

    nodes = analyze_control_flow(content, "check.py")

    kinds = [node.kind for node in nodes]

    assert kinds == [
        "if",
        "return",
        "return",
    ]


def test_nested_if():
    content = """
def check(user):
    if user:
        if user.active:
            return True
"""

    nodes = analyze_control_flow(content, "nested.py")

    assert len(nodes) == 3

    assert nodes[0].kind == "if"
    assert nodes[0].depth == 0

    assert nodes[1].kind == "if"
    assert nodes[1].depth == 1

    assert nodes[2].kind == "return"
    assert nodes[2].depth == 2


def test_for_loop():
    content = """
def process(items):
    for item in items:
        print(item)
"""

    nodes = analyze_control_flow(content, "loop.py")

    assert len(nodes) == 1
    assert nodes[0].kind == "for"
    assert nodes[0].parent == "process"
    assert nodes[0].depth == 0


def test_while_loop():
    content = """
def process(value):
    while value:
        value -= 1
"""

    nodes = analyze_control_flow(content, "loop.py")

    assert len(nodes) == 1
    assert nodes[0].kind == "while"
    assert nodes[0].parent == "process"


def test_for_with_break_and_continue():
    content = """
def process(items):
    for item in items:
        if not item:
            continue

        if item == "stop":
            break
"""

    nodes = analyze_control_flow(content, "loop.py")

    kinds = [node.kind for node in nodes]

    assert kinds == [
        "for",
        "if",
        "continue",
        "if",
        "break",
    ]

    assert nodes[0].depth == 0
    assert nodes[1].depth == 1
    assert nodes[2].depth == 2
    assert nodes[3].depth == 1
    assert nodes[4].depth == 2


def test_try_except_finally():
    content = """
def load():
    try:
        return read()
    except ValueError:
        return None
    finally:
        cleanup()
"""

    nodes = analyze_control_flow(content, "try.py")

    kinds = [node.kind for node in nodes]

    assert kinds == [
        "try",
        "return",
        "return",
    ]

    assert nodes[0].depth == 0
    assert nodes[1].depth == 1
    assert nodes[2].depth == 1


def test_try_else():
    content = """
def load():
    try:
        value = read()
    except ValueError:
        return None
    else:
        return value
"""

    nodes = analyze_control_flow(content, "try.py")

    kinds = [node.kind for node in nodes]

    assert kinds == [
        "try",
        "return",
        "return",
    ]


def test_with_statement():
    content = """
def read_file():
    with open("file.txt") as handle:
        return handle.read()
"""

    nodes = analyze_control_flow(content, "file.py")

    assert len(nodes) == 2

    assert nodes[0].kind == "with"
    assert nodes[0].parent == "read_file"

    assert nodes[1].kind == "return"
    assert nodes[1].depth == 1


def test_async_with_statement():
    content = """
async def read_file():
    async with manager() as handle:
        return await handle.read()
"""

    nodes = analyze_control_flow(content, "async_file.py")

    assert len(nodes) == 2

    assert nodes[0].kind == "with"
    assert nodes[0].parent == "read_file"

    assert nodes[1].kind == "return"


def test_async_for_loop():
    content = """
async def process(items):
    async for item in items:
        if item:
            return item
"""

    nodes = analyze_control_flow(content, "async.py")

    assert len(nodes) == 3

    assert nodes[0].kind == "for"
    assert nodes[0].parent == "process"

    assert nodes[1].kind == "if"
    assert nodes[1].depth == 1

    assert nodes[2].kind == "return"
    assert nodes[2].depth == 2


def test_raise():
    content = """
def validate(value):
    if not value:
        raise ValueError("invalid")
"""

    nodes = analyze_control_flow(content, "raise.py")

    assert len(nodes) == 2

    assert nodes[0].kind == "if"
    assert nodes[1].kind == "raise"
    assert nodes[1].depth == 1


def test_class_method_parent():
    content = """
class Service:
    def run(self, value):
        if value:
            return True
"""

    nodes = analyze_control_flow(content, "service.py")

    assert len(nodes) == 2

    assert nodes[0].kind == "if"
    assert nodes[0].parent == "run"

    assert nodes[1].kind == "return"
    assert nodes[1].parent == "run"


def test_async_function_parent():
    content = """
async def run():
    if True:
        return 1
"""

    nodes = analyze_control_flow(content, "async.py")

    assert len(nodes) == 2
    assert nodes[0].parent == "run"
    assert nodes[1].parent == "run"


def test_module_level_control_flow():
    content = """
if __name__ == "__main__":
    print("hello")
"""

    nodes = analyze_control_flow(content, "main.py")

    assert len(nodes) == 1

    assert nodes[0].kind == "if"
    assert nodes[0].parent is None
    assert nodes[0].depth == 0


def test_find_by_kind():
    content = """
def process(items):
    for item in items:
        if item:
            continue

        if item == 2:
            break
"""

    nodes = analyze_control_flow(content, "process.py")

    if_nodes = find_control_flow_by_kind(nodes, "if")
    loop_nodes = find_control_flow_by_kind(nodes, "for")

    assert len(if_nodes) == 2
    assert len(loop_nodes) == 1


def test_find_for_parent():
    content = """
def first():
    if True:
        return 1


def second():
    for item in items:
        continue
"""

    nodes = analyze_control_flow(content, "functions.py")

    first_nodes = find_control_flow_for_parent(nodes, "first")
    second_nodes = find_control_flow_for_parent(nodes, "second")

    assert len(first_nodes) == 2
    assert [node.kind for node in first_nodes] == [
        "if",
        "return",
    ]

    assert len(second_nodes) == 2
    assert [node.kind for node in second_nodes] == [
        "for",
        "continue",
    ]


def test_end_line_information():
    content = """
def process(value):
    if value:
        result = value
        return result
"""

    nodes = analyze_control_flow(content, "lines.py")

    assert nodes[0].line == 3
    assert nodes[0].end_line == 5

    assert nodes[1].line == 5
    assert nodes[1].end_line == 5


def test_syntax_error_returns_empty_list():
    content = """
def broken(
    if True:
        return
"""

    nodes = analyze_control_flow(content, "broken.py")

    assert nodes == []


def test_control_flow_node_is_structured():
    node = ControlFlowNode(
        kind="if",
        file="example.py",
        line=10,
        end_line=12,
        parent="run",
        depth=2,
    )

    assert node.kind == "if"
    assert node.file == "example.py"
    assert node.line == 10
    assert node.end_line == 12
    assert node.parent == "run"
    assert node.depth == 2


def test_build_report_empty():
    report = build_control_flow_report([])

    assert "FLY-CODER CONTROL-FLOW REPORT" in report
    assert "No control-flow constructs detected." in report


def test_build_report():
    content = """
def process(items):
    for item in items:
        if item:
            return item
"""

    nodes = analyze_control_flow(content, "process.py")

    report = build_control_flow_report(nodes)

    assert "FLY-CODER CONTROL-FLOW REPORT" in report
    assert "File: process.py" in report
    assert "Nodes: 3" in report
    assert "for: 1" in report
    assert "if: 1" in report
    assert "return: 1" in report
    assert "parent=process" in report


def test_report_contains_depth():
    content = """
def process(value):
    if value:
        if value > 10:
            return value
"""

    nodes = analyze_control_flow(content, "depth.py")

    report = build_control_flow_report(nodes)

    assert "depth=0" in report
    assert "depth=1" in report
    assert "depth=2" in report