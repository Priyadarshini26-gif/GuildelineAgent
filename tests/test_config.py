import pytest

from src.config import api_key_is_configured, load_settings


def test_defaults_load(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "5")
    monkeypatch.setenv("MAX_RETRIEVAL_ITERATIONS", "3")
    s = load_settings()
    assert s.retrieval_top_k == 5
    assert s.max_retrieval_iterations == 3


def test_invalid_limit_raises(monkeypatch):
    monkeypatch.setenv("RETRIEVAL_TOP_K", "zero")
    with pytest.raises(ValueError):
        load_settings()


def test_non_positive_limit_raises(monkeypatch):
    monkeypatch.setenv("MAX_RETRIEVAL_ITERATIONS", "0")
    with pytest.raises(ValueError):
        load_settings()


def test_placeholder_key_not_counted_as_configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "your-api-key-here")
    assert api_key_is_configured(load_settings()) is False


def test_real_looking_key_counts_as_configured(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "some-real-value")
    assert api_key_is_configured(load_settings()) is True