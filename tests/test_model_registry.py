"""Tests for ablation model registry (no GPU / checkpoint required)."""

import os

from deployment.model_registry import (
    DEFAULT_MODEL_ID,
    MODEL_CATALOG,
    list_models,
    model_path_for_id,
)


def test_catalog_ids():
    assert set(MODEL_CATALOG) == {"A0", "A1", "A4"}
    assert DEFAULT_MODEL_ID == "A4"


def test_model_path_for_id():
    p = model_path_for_id("a4")
    assert p.replace("\\", "/").endswith("outputs/ablation/A4")


def test_unknown_id_raises():
    try:
        model_path_for_id("C1")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Unknown" in str(exc)


def test_list_models_shape():
    items = list_models()
    assert [m["id"] for m in items] == ["A4", "A0", "A1"]
    for m in items:
        assert "label" in m and "description" in m and "available" in m
        assert isinstance(m["available"], bool)
