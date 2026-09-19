"""Tests for FLY-CODER test discovery."""

from flycoder.tools.testing import (
    TestDiscoveryResult,
    build_test_discovery_report,
    discover_tests,
    find_tests_by_name,
)


def test_discover_tests_finds_test_files(tmp_path):
    (tmp_path / "test_alpha.py").write_text(
        "def test_alpha():\n    pass\n",
        encoding="utf-8",
    )
    (tmp_path / "module.py").write_text(
        "value = 1\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == ["test_alpha.py"]
    assert result.count == 1


def test_discover_tests_finds_suffix_test_files(tmp_path):
    (tmp_path / "alpha_test.py").write_text(
        "def test_alpha():\n    pass\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == ["alpha_test.py"]


def test_discover_tests_finds_test_directories(tmp_path):
    tests_directory = tmp_path / "tests"
    tests_directory.mkdir()
    (tests_directory / "test_alpha.py").write_text(
        "def test_alpha():\n    pass\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.directories == ["tests"]
    assert result.files == ["tests/test_alpha.py"]


def test_discover_tests_finds_nested_test_directories(tmp_path):
    nested = tmp_path / "src" / "tests"
    nested.mkdir(parents=True)
    (nested / "test_alpha.py").write_text(
        "def test_alpha():\n    pass\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.directories == ["src/tests"]
    assert result.files == ["src/tests/test_alpha.py"]


def test_discover_tests_ignores_virtual_environment(tmp_path):
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "test_fake.py").write_text(
        "def test_fake():\n    pass\n",
        encoding="utf-8",
    )

    (tmp_path / "test_real.py").write_text(
        "def test_real():\n    pass\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == ["test_real.py"]


def test_discover_tests_ignores_pycache(tmp_path):
    ignored = tmp_path / "__pycache__"
    ignored.mkdir()
    (ignored / "test_fake.py").write_text(
        "def test_fake():\n    pass\n",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == []


def test_discover_tests_returns_sorted_paths(tmp_path):
    (tmp_path / "test_z.py").write_text("", encoding="utf-8")
    (tmp_path / "test_a.py").write_text("", encoding="utf-8")
    (tmp_path / "test_m.py").write_text("", encoding="utf-8")

    result = discover_tests(tmp_path)

    assert result.files == [
        "test_a.py",
        "test_m.py",
        "test_z.py",
    ]


def test_discover_tests_uses_posix_paths(tmp_path):
    nested = tmp_path / "package" / "tests"
    nested.mkdir(parents=True)

    (nested / "test_example.py").write_text(
        "",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == ["package/tests/test_example.py"]


def test_discover_tests_missing_root(tmp_path):
    result = discover_tests(tmp_path / "missing")

    assert result.files == []
    assert result.directories == []
    assert result.count == 0


def test_find_tests_by_name(tmp_path):
    (tmp_path / "test_auth.py").write_text("", encoding="utf-8")
    (tmp_path / "test_database.py").write_text("", encoding="utf-8")
    (tmp_path / "test_ui.py").write_text("", encoding="utf-8")

    result = discover_tests(tmp_path)

    assert find_tests_by_name(result, "auth") == ["test_auth.py"]


def test_find_tests_by_name_is_case_insensitive(tmp_path):
    (tmp_path / "test_Authentication.py").write_text(
        "",
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert find_tests_by_name(result, "AUTH") == [
        "test_Authentication.py"
    ]


def test_find_tests_by_name_empty_query(tmp_path):
    (tmp_path / "test_example.py").write_text("", encoding="utf-8")

    result = discover_tests(tmp_path)

    assert find_tests_by_name(result, "") == []


def test_build_test_discovery_report():
    result = TestDiscoveryResult(
        files=["tests/test_a.py"],
        directories=["tests"],
    )

    report = build_test_discovery_report(result)

    assert report == {
        "test_files": ["tests/test_a.py"],
        "test_directories": ["tests"],
        "test_count": 1,
    }


def test_discovery_does_not_execute_tests(tmp_path):
    test_file = tmp_path / "test_side_effect.py"
    test_file.write_text(
        'raise RuntimeError("Tests were executed")\n',
        encoding="utf-8",
    )

    result = discover_tests(tmp_path)

    assert result.files == ["test_side_effect.py"]
