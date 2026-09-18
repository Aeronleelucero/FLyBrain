from flycoder.tools.data_flow import (
    DataFlowInfo,
    analyze_data_flow,
    build_data_flow_report,
    find_data_flow_by_kind,
    find_data_flow_by_source,
    find_data_flow_by_target,
    find_data_flow_for_parent,
)


def test_empty_source():
    flows = analyze_data_flow("", "empty.py")

    assert flows == []


def test_parameter_flow():
    content = """
def calculate(price):
    return price
"""

    flows = analyze_data_flow(content, "calculate.py")

    parameters = find_data_flow_by_kind(
        flows,
        "parameter",
    )

    assert len(parameters) == 1
    assert parameters[0].source == "<input>"
    assert parameters[0].target == "price"
    assert parameters[0].parent == "calculate"


def test_assignment_flow():
    content = """
def calculate(price):
    tax = price * 0.12
"""

    flows = analyze_data_flow(content, "calculate.py")

    assignments = find_data_flow_by_kind(
        flows,
        "assignment",
    )

    assert any(
        flow.source == "price"
        and flow.target == "tax"
        for flow in assignments
    )


def test_multiple_sources_to_assignment():
    content = """
def calculate(price, tax):
    total = price + tax
"""

    flows = analyze_data_flow(content, "calculate.py")

    assignments = find_data_flow_by_kind(
        flows,
        "assignment",
    )

    assert any(
        flow.source == "price"
        and flow.target == "total"
        for flow in assignments
    )

    assert any(
        flow.source == "tax"
        and flow.target == "total"
        for flow in assignments
    )


def test_return_flow():
    content = """
def calculate(total):
    return total
"""

    flows = analyze_data_flow(content, "calculate.py")

    returns = find_data_flow_by_kind(
        flows,
        "return",
    )

    assert len(returns) == 1
    assert returns[0].source == "total"
    assert returns[0].target == "<return>"
    assert returns[0].parent == "calculate"


def test_call_input_flow():
    content = """
def calculate(price):
    return round(price)
"""

    flows = analyze_data_flow(content, "calculate.py")

    inputs = find_data_flow_by_kind(
        flows,
        "call_input",
    )

    assert any(
        flow.source == "price"
        and flow.target == "round()"
        for flow in inputs
    )


def test_call_result_flow():
    content = """
def calculate(price):
    total = calculate_tax(price)
    return total
"""

    flows = analyze_data_flow(content, "calculate.py")

    results = find_data_flow_by_kind(
        flows,
        "call_result",
    )

    assert any(
        flow.source == "calculate_tax()"
        and flow.target == "total"
        for flow in results
    )


def test_call_result_and_input_flow():
    content = """
def calculate(price):
    total = calculate_tax(price)
"""

    flows = analyze_data_flow(content, "calculate.py")

    assert any(
        flow.source == "price"
        and flow.target == "calculate_tax()"
        and flow.kind == "call_input"
        for flow in flows
    )

    assert any(
        flow.source == "calculate_tax()"
        and flow.target == "total"
        and flow.kind == "call_result"
        for flow in flows
    )


def test_attribute_source():
    content = """
def get_name(user):
    name = user.name
    return name
"""

    flows = analyze_data_flow(content, "user.py")

    assert any(
        flow.source == "user.name"
        and flow.target == "name"
        for flow in flows
    )


def test_attribute_call_input():
    content = """
def save(user):
    result = user.save()
    return result
"""

    flows = analyze_data_flow(content, "user.py")

    assert any(
        flow.source == "user"
        and flow.target == "user.save()"
        and flow.kind == "call_input"
        for flow in flows
    )


def test_augmented_assignment():
    content = """
def increment(count):
    count += 1
    return count
"""

    flows = analyze_data_flow(content, "counter.py")

    updates = find_data_flow_by_kind(
        flows,
        "update",
    )

    assert any(
        flow.source == "count"
        and flow.target == "count"
        for flow in updates
    )


def test_augmented_assignment_with_source():
    content = """
def add(total, value):
    total += value
"""

    flows = analyze_data_flow(content, "total.py")

    updates = find_data_flow_by_kind(
        flows,
        "update",
    )

    assert any(
        flow.source == "value"
        and flow.target == "total"
        for flow in updates
    )


def test_multiple_assignment_targets():
    content = """
def process(value):
    first = second = value
"""

    flows = analyze_data_flow(content, "process.py")

    assignments = find_data_flow_by_kind(
        flows,
        "assignment",
    )

    assert any(
        flow.source == "value"
        and flow.target == "first"
        for flow in assignments
    )

    assert any(
        flow.source == "value"
        and flow.target == "second"
        for flow in assignments
    )


def test_tuple_unpacking():
    content = """
def process():
    first, second = get_values()
"""

    flows = analyze_data_flow(content, "unpack.py")

    results = find_data_flow_by_kind(
        flows,
        "call_result",
    )

    assert any(
        flow.source == "get_values()"
        and flow.target == "first"
        for flow in results
    )

    assert any(
        flow.source == "get_values()"
        and flow.target == "second"
        for flow in results
    )


def test_annotated_assignment():
    content = """
def process(value):
    result: int = value
"""

    flows = analyze_data_flow(content, "annotated.py")

    assert any(
        flow.source == "value"
        and flow.target == "result"
        and flow.kind == "assignment"
        for flow in flows
    )


def test_named_expression():
    content = """
def process(value):
    if (result := value):
        return result
"""

    flows = analyze_data_flow(content, "walrus.py")

    assert any(
        flow.source == "value"
        and flow.target == "result"
        and flow.kind == "assignment"
        for flow in flows
    )


def test_keyword_call_input():
    content = """
def process(value):
    return save(data=value)
"""

    flows = analyze_data_flow(content, "keyword.py")

    assert any(
        flow.source == "value"
        and flow.target == "save()"
        and flow.kind == "call_input"
        for flow in flows
    )


def test_varargs_parameter():
    content = """
def process(*items):
    return items
"""

    flows = analyze_data_flow(content, "args.py")

    parameters = find_data_flow_by_kind(
        flows,
        "parameter",
    )

    assert any(
        flow.target == "items"
        for flow in parameters
    )


def test_kwargs_parameter():
    content = """
def process(**options):
    return options
"""

    flows = analyze_data_flow(content, "kwargs.py")

    parameters = find_data_flow_by_kind(
        flows,
        "parameter",
    )

    assert any(
        flow.target == "options"
        for flow in parameters
    )


def test_method_parent():
    content = """
class Service:
    def run(self, value):
        result = calculate(value)
        return result
"""

    flows = analyze_data_flow(content, "service.py")

    method_flows = find_data_flow_for_parent(
        flows,
        "run",
    )

    assert method_flows
    assert all(
        flow.parent == "run"
        for flow in method_flows
    )


def test_find_by_source():
    content = """
def process(value):
    first = value
    second = value
"""

    flows = analyze_data_flow(content, "process.py")

    matches = find_data_flow_by_source(
        flows,
        "value",
    )

    assert len(matches) >= 2


def test_find_by_target():
    content = """
def process(value, other):
    result = value + other
"""

    flows = analyze_data_flow(content, "process.py")

    matches = find_data_flow_by_target(
        flows,
        "result",
    )

    assert len(matches) == 2


def test_metadata():
    content = """
def process(value):
    result = value
"""

    flows = analyze_data_flow(content, "metadata.py")

    assignment = next(
        flow
        for flow in flows
        if flow.kind == "assignment"
    )

    assert assignment.file == "metadata.py"
    assert assignment.line == 3
    assert assignment.parent == "process"


def test_structured_data_flow():
    flow = DataFlowInfo(
        source="price",
        target="tax",
        kind="assignment",
        file="calculate.py",
        line=10,
        parent="calculate",
    )

    assert flow.source == "price"
    assert flow.target == "tax"
    assert flow.kind == "assignment"
    assert flow.file == "calculate.py"
    assert flow.line == 10
    assert flow.parent == "calculate"


def test_syntax_error_returns_empty_list():
    content = """
def broken(
    value =
"""

    flows = analyze_data_flow(
        content,
        "broken.py",
    )

    assert flows == []


def test_empty_report():
    report = build_data_flow_report([])

    assert "FLY-CODER DATA-FLOW REPORT" in report
    assert "No data-flow relationships detected." in report


def test_data_flow_report():
    content = """
def calculate(price):
    tax = calculate_tax(price)
    return tax
"""

    flows = analyze_data_flow(
        content,
        "calculate.py",
    )

    report = build_data_flow_report(flows)

    assert "FLY-CODER DATA-FLOW REPORT" in report
    assert "File: calculate.py" in report
    assert "calculate_tax()" in report
    assert "price" in report
    assert "tax" in report
    assert "<return>" in report