"""Tests for ablation model registry (no GPU / checkpoint required)."""

import json
import os

from deployment.model_registry import (
    DEFAULT_MODEL_ID,
    MODEL_CATALOG,
    find_checkpoint_dir,
    find_legacy_a1_checkpoint,
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


def test_find_checkpoint_prefers_root(tmp_path):
    run = tmp_path / "A0"
    run.mkdir()
    (run / "config.json").write_text("{}", encoding="utf-8")
    assert find_checkpoint_dir(str(run)) == str(run)


def test_find_checkpoint_prefers_seed_42(tmp_path):
    run = tmp_path / "A4"
    run.mkdir()
    s41 = run / "seed_41"
    s42 = run / "seed_42"
    s41.mkdir()
    s42.mkdir()
    (s41 / "config.json").write_text("{}", encoding="utf-8")
    (s42 / "config.json").write_text("{}", encoding="utf-8")
    assert find_checkpoint_dir(str(run)) == str(s42)


def test_find_checkpoint_falls_back_to_lowest_seed(tmp_path):
    run = tmp_path / "A4"
    run.mkdir()
    s45 = run / "seed_45"
    s45.mkdir()
    (s45 / "config.json").write_text("{}", encoding="utf-8")
    assert find_checkpoint_dir(str(run)) == str(s45)


def test_find_checkpoint_uses_highest_checkpoint_step(tmp_path):
    run = tmp_path / "legacy"
    c1 = run / "checkpoint-40068"
    c2 = run / "checkpoint-60102"
    c1.mkdir(parents=True)
    c2.mkdir(parents=True)
    (c1 / "config.json").write_text("{}", encoding="utf-8")
    (c2 / "config.json").write_text("{}", encoding="utf-8")
    assert find_checkpoint_dir(str(run)) == str(c2)


def test_resolve_uses_seed_subdir(tmp_path, monkeypatch):
    import deployment.model_registry as mr

    ablation = tmp_path / "ablation"
    a4 = ablation / "A4" / "seed_42"
    a4.mkdir(parents=True)
    (a4 / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(mr, "ablation_dir", lambda: str(ablation))
    resolved = mr.resolve_model_id("A4")
    assert resolved["model_id"] == "A4"
    assert resolved["model_path"] == str(a4)


def test_legacy_a1_discovers_fear_aug_run(tmp_path, monkeypatch):
    import deployment.model_registry as mr

    outputs = tmp_path / "outputs"
    # Misnamed allAug run (must be ignored)
    bad = outputs / "full_8label_aug_20260708_1436"
    bad_ckpt = bad / "checkpoint-1"
    bad_ckpt.mkdir(parents=True)
    (bad_ckpt / "config.json").write_text("{}", encoding="utf-8")
    (bad / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": "/x/Combined_Labeled_Dataset_with_allAug.csv",
                "dataset_size": 412476,
            }
        ),
        encoding="utf-8",
    )
    # Real A1
    good = outputs / "full_8label_aug_20260423_0536"
    good_ckpt = good / "checkpoint-60102"
    good_ckpt.mkdir(parents=True)
    (good_ckpt / "config.json").write_text("{}", encoding="utf-8")
    (good / "training_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": "/x/Combined_Labeled_Dataset_with_fearAug.csv",
                "dataset_size": 400691,
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mr, "outputs_dir", lambda: str(outputs))
    monkeypatch.setattr(mr, "ablation_dir", lambda: str(tmp_path / "ablation"))
    found = find_legacy_a1_checkpoint()
    assert found == str(good_ckpt)
    resolved = mr.resolve_model_id("A1")
    assert resolved["model_path"] == str(good_ckpt)
