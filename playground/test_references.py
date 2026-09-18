from flycoder.tools.references import (
    ReferenceInfo,
    analyze_references,
    build_reference_report,
    find_references_by_kind,
    find_references_by_name,
    find_references_for_parent,
)


def test_empty_source():
    references = analyze_references("", "empty.py")

    assert references == []


def test_simple_name_read():
    content = """
def run(value):
    return value
"""

    references = analyze_references(content, "example.py")

    value_reads = [
        reference
        for reference in references
        if reference.name == "value"
        and reference.kind == "read"
    ]

    assert len(value_reads) == 1
    assert value_reads[0].parent == "run"


def test_assignment_reference():
    content = """
def run():
    result = 10
    return result
"""

    references = analyze_references(content, "example.py")

    assignments = find_references_by_kind(
        references,
        "assignment",
    )

    assert any(
        reference.name == "result"
        for reference in assignments
    )


def test_read_reference():
    content = """
def run():
    result = 10
    return result
"""

    references = analyze_references(content, "example.py")

    reads = find_references_by_kind(
        references,
        "read",
    )

    assert any(
        reference.name == "result"
        for reference in reads
    )


def test_function_call():
    content = """
def run(value):
    return calculate(value)
"""

    references = analyze_references(content, "example.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    assert any(
        reference.name == "calculate"
        for reference in calls
    )


def test_uppercase_call_is_instantiation():
    content = """
def run():
    service = UserService()
    return service
"""

    references = analyze_references(content, "example.py")

    instantiations = find_references_by_kind(
        references,
        "instantiation",
    )

    assert len(instantiations) == 1
    assert instantiations[0].name == "UserService"
    assert instantiations[0].parent == "run"


def test_attribute_reference():
    content = """
def run(user):
    return user.name
"""

    references = analyze_references(content, "example.py")

    assert any(
        reference.name == "user.name"
        and reference.kind == "read"
        for reference in references
    )


def test_attribute_call():
    content = """
def run(user):
    return user.save()
"""

    references = analyze_references(content, "example.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    assert any(
        reference.name == "user.save"
        for reference in calls
    )


def test_nested_attribute_reference():
    content = """
def run(service):
    return service.api.send()
"""

    references = analyze_references(content, "example.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    assert any(
        reference.name == "service.api.send"
        for reference in calls
    )


def test_nested_attribute_read():
    content = """
def run(service):
    return service.api.value
"""

    references = analyze_references(content, "example.py")

    reads = find_references_by_kind(
        references,
        "read",
    )

    assert any(
        reference.name == "service.api.value"
        for reference in reads
    )


def test_method_parent():
    content = """
class Service:
    def run(self, user):
        return user.save()
"""

    references = analyze_references(content, "service.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    assert any(
        reference.name == "user.save"
        and reference.parent == "run"
        for reference in calls
    )


def test_async_function_parent():
    content = """
async def run(service):
    return await service.fetch()
"""

    references = analyze_references(content, "async.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    assert any(
        reference.name == "service.fetch"
        and reference.parent == "run"
        for reference in calls
    )


def test_multiple_calls():
    content = """
def run(service):
    service.start()
    service.process()
    service.stop()
"""

    references = analyze_references(content, "service.py")

    calls = find_references_by_kind(
        references,
        "call",
    )

    names = [reference.name for reference in calls]

    assert "service.start" in names
    assert "service.process" in names
    assert "service.stop" in names


def test_function_arguments_are_referenced():
    content = """
def calculate(price, tax):
    return price + tax
"""

    references = analyze_references(content, "calculate.py")

    names = {
        reference.name
        for reference in references
        if reference.kind == "read"
    }

    assert "price" in names
    assert "tax" in names


def test_if_condition_reference():
    content = """
def check(user):
    if user.active:
        return True
"""

    references = analyze_references(content, "check.py")

    assert any(
        reference.name == "user.active"
        and reference.kind == "read"
        for reference in references
    )


def test_loop_reference():
    content = """
def process(items):
    for item in items:
        print(item)
"""

    references = analyze_references(content, "loop.py")

    names = [
        reference.name
        for reference in references
    ]

    assert "items" in names
    assert "item" in names
    assert "print" in names


def test_delete_reference():
    content = """
def remove(data):
    del data.value
"""

    references = analyze_references(content, "delete.py")

    deletes = find_references_by_kind(
        references,
        "delete",
    )

    assert any(
        reference.name == "data.value"
        for reference in deletes
    )


def test_find_by_name():
    content = """
def run(value):
    result = value
    return value
"""

    references = analyze_references(content, "example.py")

    matches = find_references_by_name(
        references,
        "value",
    )

    assert len(matches) == 2


def test_find_by_parent():
    content = """
def first(value):
    return calculate(value)


def second(value):
    return save(value)
"""

    references = analyze_references(content, "example.py")

    first = find_references_for_parent(
        references,
        "first",
    )

    second = find_references_for_parent(
        references,
        "second",
    )

    assert any(
        reference.name == "calculate"
        for reference in first
    )

    assert any(
        reference.name == "save"
        for reference in second
    )


def test_reference_metadata():
    content = """
def run():
    value = 10
"""

    references = analyze_references(content, "metadata.py")

    value = next(
        reference
        for reference in references
        if reference.name == "value"
    )

    assert value.file == "metadata.py"
    assert value.line == 3
    assert value.end_line == 3
    assert value.parent == "run"


def test_reference_info_is_structured():
    reference = ReferenceInfo(
        name="UserService",
        kind="instantiation",
        file="service.py",
        line=10,
        end_line=10,
        parent="run",
    )

    assert reference.name == "UserService"
    assert reference.kind == "instantiation"
    assert reference.file == "service.py"
    assert reference.line == 10
    assert reference.end_line == 10
    assert reference.parent == "run"


def test_syntax_error_returns_empty_list():
    content = """
def broken(
    value =
"""

    references = analyze_references(
        content,
        "broken.py",
    )

    assert references == []


def test_build_empty_report():
    report = build_reference_report([])

    assert "FLY-CODER SYMBOL REFERENCE REPORT" in report
    assert "No symbol references detected." in report


def test_build_reference_report():
    content = """
def run(service):
    service.start()
    value = service.fetch()
    return value
"""

    references = analyze_references(
        content,
        "service.py",
    )

    report = build_reference_report(references)

    assert "FLY-CODER SYMBOL REFERENCE REPORT" in report
    assert "File: service.py" in report
    assert "References:" in report
    assert "service.start" in report
    assert "service.fetch" in report
    assert "parent=run" in report