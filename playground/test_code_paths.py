from flycoder.tools.code_paths import (
    CodePath,
    analyze_code_paths,
    build_code_path_report,
    find_code_paths_by_kind,
    find_code_paths_by_name,
    find_code_paths_for_file,
)


def test_empty_source():
    assert analyze_code_paths("") == []


def test_simple_function_has_one_path():
    source = """
def hello():
    return "hello"
"""

    results = analyze_code_paths(source, "example.py")

    assert len(results) == 1

    result = results[0]

    assert result.name == "hello"
    assert result.kind == "function"
    assert result.file == "example.py"
    assert result.start_line == 2
    assert result.end_line == 3
    assert result.path_count == 1
    assert result.exit_points == 1
    assert result.returns == 1
    assert result.raises == 0


def test_function_without_explicit_exit():
    source = """
def work():
    value = 10
"""

    result = analyze_code_paths(source)[0]

    assert result.path_count == 1
    assert result.exit_points == 0
    assert result.returns == 0
    assert result.raises == 0


def test_if_creates_two_paths():
    source = """
def check(value):
    if value:
        return 1
    return 0
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 1
    assert result.path_count == 2
    assert result.returns == 2
    assert result.exit_points == 2


def test_if_else_has_two_paths():
    source = """
def check(value):
    if value:
        return 1
    else:
        return 2
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 1
    assert result.path_count == 2
    assert result.returns == 2


def test_multiple_branches():
    source = """
def check(a, b):
    if a:
        return 1

    if b:
        return 2

    return 3
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 2
    assert result.path_count == 3
    assert result.returns == 3


def test_nested_branches():
    source = """
def check(a, b):
    if a:
        if b:
            return 1
    return 0
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 2
    assert result.path_count == 3
    assert result.max_nesting == 2


def test_for_loop_adds_path():
    source = """
def process(items):
    for item in items:
        print(item)
"""

    result = analyze_code_paths(source)[0]

    assert result.loops == 1
    assert result.path_count == 2


def test_while_loop_adds_path():
    source = """
def process(value):
    while value > 0:
        value -= 1
"""

    result = analyze_code_paths(source)[0]

    assert result.loops == 1
    assert result.path_count == 2


def test_nested_loops():
    source = """
def process(items):
    for item in items:
        while item:
            item -= 1
"""

    result = analyze_code_paths(source)[0]

    assert result.loops == 2
    assert result.path_count == 3
    assert result.max_nesting == 2


def test_async_for():
    source = """
async def process(items):
    async for item in items:
        print(item)
"""

    result = analyze_code_paths(source)[0]

    assert result.kind == "async_function"
    assert result.loops == 1
    assert result.path_count == 2


def test_try_except_creates_exception_path():
    source = """
def load():
    try:
        return read()
    except ValueError:
        return None
"""

    result = analyze_code_paths(source)[0]

    assert result.exceptions == 1
    assert result.path_count == 2
    assert result.returns == 2
    assert result.exit_points == 2


def test_multiple_exception_handlers():
    source = """
def load():
    try:
        return read()
    except ValueError:
        return None
    except TypeError:
        return None
"""

    result = analyze_code_paths(source)[0]

    assert result.exceptions == 2
    assert result.path_count == 3
    assert result.returns == 3


def test_raise_is_an_exit_point():
    source = """
def fail():
    raise ValueError("bad")
"""

    result = analyze_code_paths(source)[0]

    assert result.raises == 1
    assert result.exit_points == 1
    assert result.returns == 0
    assert result.path_count == 1


def test_return_and_raise():
    source = """
def process(value):
    if value:
        return value
    raise ValueError("bad")
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 1
    assert result.returns == 1
    assert result.raises == 1
    assert result.exit_points == 2
    assert result.path_count == 2


def test_conditional_expression_adds_branch():
    source = """
def check(value):
    return 1 if value else 0
"""

    result = analyze_code_paths(source)[0]

    assert result.branches == 1
    assert result.path_count == 2


def test_with_statement_does_not_create_alternative_path():
    source = """
def read():
    with open("test.txt") as handle:
        return handle.read()
"""

    result = analyze_code_paths(source)[0]

    assert result.path_count == 1
    assert result.exit_points == 1
    assert result.max_nesting == 1


def test_async_with_statement():
    source = """
async def read():
    async with resource() as handle:
        return await handle.read()
"""

    result = analyze_code_paths(source)[0]

    assert result.kind == "async_function"
    assert result.path_count == 1
    assert result.max_nesting == 1


def test_nested_function_is_analyzed():
    source = """
def outer():
    def inner():
        if True:
            return 1
        return 0

    return inner()
"""

    results = analyze_code_paths(source)

    assert len(results) == 2

    assert results[0].name == "outer"
    assert results[0].returns == 1

    assert results[1].name == "inner"
    assert results[1].branches == 1
    assert results[1].path_count == 2


def test_async_function():
    source = """
async def fetch(url):
    if url:
        return await request(url)
    return None
"""

    result = analyze_code_paths(source)[0]

    assert result.name == "fetch"
    assert result.kind == "async_function"
    assert result.branches == 1
    assert result.path_count == 2


def test_syntax_error_returns_empty():
    source = """
def broken(:
    pass
"""

    assert analyze_code_paths(source, "broken.py") == []


def test_find_by_name():
    source = """
def first():
    return 1

def second():
    if True:
        return 2
    return 3
"""

    results = analyze_code_paths(source)

    matches = find_code_paths_by_name(results, "second")

    assert len(matches) == 1
    assert matches[0].name == "second"
    assert matches[0].path_count == 2


def test_find_by_kind():
    source = """
def normal():
    return 1

async def asynchronous():
    return 2
"""

    results = analyze_code_paths(source)

    normal = find_code_paths_by_kind(results, "function")
    asynchronous = find_code_paths_by_kind(results, "async_function")

    assert len(normal) == 1
    assert normal[0].name == "normal"

    assert len(asynchronous) == 1
    assert asynchronous[0].name == "asynchronous"


def test_find_for_file():
    source_a = """
def first():
    return 1
"""

    source_b = """
def second():
    return 2
"""

    results = (
        analyze_code_paths(source_a, "a.py")
        + analyze_code_paths(source_b, "b.py")
    )

    matches = find_code_paths_for_file(results, "b.py")

    assert len(matches) == 1
    assert matches[0].name == "second"


def test_code_path_is_immutable():
    result = CodePath(
        name="test",
        kind="function",
        file="test.py",
        start_line=1,
        end_line=2,
        path_count=1,
        exit_points=1,
        branches=0,
        loops=0,
        exceptions=0,
        returns=1,
        raises=0,
        max_nesting=0,
    )

    try:
        result.name = "changed"
        changed = True
    except AttributeError:
        changed = False

    assert changed is False


def test_build_code_path_report():
    source = """
def calculate(value):
    if value:
        return helper(value)
    return 0
"""

    results = analyze_code_paths(source, "calculator.py")

    report = build_code_path_report(results)

    assert "FLY-CODER CODE PATH REPORT" in report
    assert "Functions analyzed: 1" in report
    assert "calculate" in report
    assert "calculator.py" in report
    assert "Estimated paths: 2" in report
    assert "Exit points: 2" in report
    assert "Branches: 1" in report


def test_empty_code_path_report():
    report = build_code_path_report([])

    assert "FLY-CODER CODE PATH REPORT" in report
    assert "Functions analyzed: 0" in report
    assert "No functions or methods found." in report