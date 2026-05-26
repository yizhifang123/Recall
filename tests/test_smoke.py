"""Smoke test — ensures the package is importable and exposes a version."""


def test_package_importable() -> None:
    import recall

    assert recall.__version__ == "0.0.1"
