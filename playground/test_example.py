from example import main


def test_main_exists():
    assert callable(main)


def test_intentional_failure():
    assert True, "Intentional failure for testing"
