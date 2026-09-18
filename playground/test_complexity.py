from flycoder.tools.complexity import (
    ComplexityInfo,
    analyze_complexity,
    build_complexity_report,
    find_complexity_by_kind,
    find_complexity_by_name,
)


def test_empty_source():
    assert analyze_complexity("") == []


def test_simple_function():
    source = """
def hello():
    return "hello"
"""

    results = analyze_complexity(source, "example.py")

    assert len(results) == 1

    result = results[0]

    assert result.name == "hello"
    assert result.kind == "function"
    assert result.file == "example.py"
    assert result.line == 2
    assert result.end_line == 3
    assert result.lines == 2
    assert result.parameters == 0
    assert result.branches == 0
    assert result.loops == 0
    assert result.returns == 1
    assert result.exceptions == 0
    assert result.calls == 0
    assert result.max_nesting == 0
    assert result.cyclomatic_complexity == 1


def test_function_parameters():
    source = """
def calculate(a, b, c=10, *args, option=True, **kwargs):
    return a + b + c
"""

    results = analyze_complexity(source)

    assert len(results) == 1
    assert results[0].parameters == 6


def test_if_branch():
    source = """
def check(value):
    if value > 10:
        return True
    return False
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 1
    assert result.cyclomatic_complexity == 2


def test_if_else_only_counts_one_branch():
    source = """
def check(value):
    if value:
        return 1
    else:
        return 2
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 1
    assert result.cyclomatic_complexity == 2


def test_nested_if():
    source = """
def check(a, b):
    if a:
        if b:
            return 1
    return 0
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 2
    assert result.max_nesting == 2
    assert result.cyclomatic_complexity == 3


def test_for_loop():
    source = """
def process(items):
    for item in items:
        print(item)
"""

    result = analyze_complexity(source)[0]

    assert result.loops == 1
    assert result.calls == 1
    assert result.cyclomatic_complexity == 2


def test_while_loop():
    source = """
def process(value):
    while value > 0:
        value -= 1
"""

    result = analyze_complexity(source)[0]

    assert result.loops == 1
    assert result.cyclomatic_complexity == 2


def test_async_for_loop():
    source = """
async def process(items):
    async for item in items:
        print(item)
"""

    result = analyze_complexity(source)[0]

    assert result.kind == "async_function"
    assert result.loops == 1
    assert result.cyclomatic_complexity == 2


def test_try_except():
    source = """
def load():
    try:
        return read()
    except ValueError:
        return None
"""

    result = analyze_complexity(source)[0]

    assert result.exceptions == 1
    assert result.calls == 1
    assert result.returns == 2
    assert result.cyclomatic_complexity == 2


def test_multiple_except_handlers():
    source = """
def load():
    try:
        return read()
    except ValueError:
        return None
    except TypeError:
        return None
"""

    result = analyze_complexity(source)[0]

    assert result.exceptions == 2
    assert result.cyclomatic_complexity == 3


def test_with_statement_adds_nesting():
    source = """
def read_file():
    with open("test.txt") as handle:
        return handle.read()
"""

    result = analyze_complexity(source)[0]

    assert result.calls == 2
    assert result.max_nesting == 1


def test_async_with_statement():
    source = """
async def read_file():
    async with get_resource() as resource:
        return await resource.read()
"""

    result = analyze_complexity(source)[0]

    assert result.kind == "async_function"
    assert result.calls == 2
    assert result.max_nesting == 1


def test_boolean_expression_adds_branches():
    source = """
def check(a, b, c):
    return a and b and c
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 2
    assert result.cyclomatic_complexity == 3


def test_or_expression_adds_branches():
    source = """
def check(a, b):
    return a or b
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 1
    assert result.cyclomatic_complexity == 2


def test_conditional_expression_adds_branch():
    source = """
def check(value):
    return "yes" if value else "no"
"""

    result = analyze_complexity(source)[0]

    assert result.branches == 1
    assert result.cyclomatic_complexity == 2


def test_calls_are_counted():
    source = """
def run():
    first()
    second()
    obj.third()
"""

    result = analyze_complexity(source)[0]

    assert result.calls == 3


def test_multiple_returns():
    source = """
def check(value):
    if value:
        return 1
    if value > 10:
        return 2
    return 3
"""

    result = analyze_complexity(source)[0]

    assert result.returns == 3


def test_nested_loops():
    source = """
def process(items):
    for item in items:
        while item:
            item -= 1
"""

    result = analyze_complexity(source)[0]

    assert result.loops == 2
    assert result.max_nesting == 2
    assert result.cyclomatic_complexity == 3


def test_function_and_nested_function():
    source = """
def outer():
    def inner():
        return 1

    return inner()
"""

    results = analyze_complexity(source)

    assert len(results) == 2
    assert results[0].name == "outer"
    assert results[1].name == "inner"


def test_async_function():
    source = """
async def fetch(url):
    return await request(url)
"""

    result = analyze_complexity(source)[0]

    assert result.name == "fetch"
    assert result.kind == "async_function"
    assert result.parameters == 1
    assert result.calls == 1


def test_syntax_error_returns_empty():
    source = """
def broken(:
    pass
"""

    assert analyze_complexity(source, "broken.py") == []


def test_find_by_name():
    source = """
def first():
    return 1

def second():
    return 2
"""

    results = analyze_complexity(source)

    matches = find_complexity_by_name(results, "second")

    assert len(matches) == 1
    assert matches[0].name == "second"


def test_find_by_kind():
    source = """
def normal():
    return 1

async def async_version():
    return 2
"""

    results = analyze_complexity(source)

    normal = find_complexity_by_kind(results, "function")
    async_functions = find_complexity_by_kind(results, "async_function")

    assert len(normal) == 1
    assert normal[0].name == "normal"

    assert len(async_functions) == 1
    assert async_functions[0].name == "async_version"


def test_complexity_info_is_immutable():
    result = ComplexityInfo(
        name="test",
        kind="function",
        file="test.py",
        line=1,
        end_line=2,
        lines=2,
        parameters=0,
        branches=0,
        loops=0,
        returns=1,
        exceptions=0,
        calls=0,
        max_nesting=0,
        cyclomatic_complexity=1,
    )

    try:
        result.name = "changed"
        changed = True
    except AttributeError:
        changed = False

    assert changed is False


def test_build_complexity_report():
    source = """
def calculate(value):
    if value:
        return helper(value)
    return 0
"""

    results = analyze_complexity(source, "calculator.py")

    report = build_complexity_report(results)

    assert "FLY-CODER COMPLEXITY REPORT" in report
    assert "Functions analyzed: 1" in report
    assert "calculate" in report
    assert "calculator.py" in report
    assert "Branches: 1" in report
    assert "Calls: 1" in report
    assert "Cyclomatic complexity: 2" in report


def test_empty_complexity_report():
    report = build_complexity_report([])

    assert "FLY-CODER COMPLEXITY REPORT" in report
    assert "Functions analyzed: 0" in report
    assert "No functions or methods found." in report