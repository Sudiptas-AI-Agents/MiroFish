import pytest

from app.config import Config


@pytest.fixture(autouse=True)
def _default_to_zep_backend(monkeypatch):
    """The upstream suite exercises Zep; the Graphiti tests opt in explicitly."""

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
