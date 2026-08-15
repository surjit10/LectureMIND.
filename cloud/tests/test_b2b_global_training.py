# cloud/tests/test_b2b_global_training.py
# Feature 2 — Global reranker orchestration (scripts/train_global_reranker.py).
# Tests the merge/prepare/install steps without a GPU: training is exercised
# through the existing trainer_fn injection (no real fit()).

import json
from pathlib import Path

import pytest

from scripts.train_global_reranker import (
    collect_and_merge_triplets,
    install_global_model,
    prepare_train_dir,
    train_global,
)
from config import CloudSettings


def _make_package(tmp_path: Path, lecture_id: str, triplets) -> Path:
    d = tmp_path / lecture_id
    d.mkdir()
    (d / "triplets.json").write_text(json.dumps(triplets), encoding="utf-8")
    return d


def _triplet(q: str, pos: str, neg: str) -> dict:
    return {"query": q, "positive": pos, "negative": neg}


class TestCollectAndMerge:
    def test_merges_across_lectures(self, tmp_path):
        _make_package(tmp_path, "lecture_a", [_triplet("q1", "p1", "n1")] * 60)
        _make_package(tmp_path, "lecture_b", [_triplet("q2", "p2", "n2")] * 40)
        merged = collect_and_merge_triplets(tmp_path, min_triplets=50)
        assert len(merged) == 60  # lecture_b (40) skipped below threshold

    def test_min_triplets_zero_includes_all(self, tmp_path):
        _make_package(tmp_path, "lecture_a", [_triplet("q", "p", "n")])
        merged = collect_and_merge_triplets(tmp_path, min_triplets=0)
        assert len(merged) == 1

    def test_explicit_lecture_list(self, tmp_path):
        _make_package(tmp_path, "lecture_a", [_triplet("q", "p", "n")] * 50)
        _make_package(tmp_path, "lecture_b", [_triplet("q", "p", "n")] * 50)
        merged = collect_and_merge_triplets(tmp_path, lecture_ids=["lecture_a"], min_triplets=0)
        assert len(merged) == 50

    def test_malformed_triplets_dropped(self, tmp_path):
        d = tmp_path / "lecture_x"
        d.mkdir()
        (d / "triplets.json").write_text(
            json.dumps([_triplet("q", "p", "n"), {"bad": "shape"}]), encoding="utf-8"
        )
        merged = collect_and_merge_triplets(tmp_path, min_triplets=0)
        assert len(merged) == 1

    def test_missing_packages_dir_is_empty(self, tmp_path):
        assert collect_and_merge_triplets(tmp_path / "nope") == []


class TestPrepareAndTrain:
    def test_prepare_writes_triplets(self, tmp_path, monkeypatch):
        settings = CloudSettings(LECTURE_OUTPUT_DIR=str(tmp_path / "out" / "{lecture_id}/"))
        monkeypatch.chdir(tmp_path)
        train_dir = prepare_train_dir("train1", [_triplet("q", "p", "n")], settings)
        assert (train_dir / "triplets.json").exists()

    def test_train_global_reuses_injected_trainer(self, tmp_path, monkeypatch):
        settings = CloudSettings(LECTURE_OUTPUT_DIR=str(tmp_path / "out" / "{lecture_id}/"))
        monkeypatch.chdir(tmp_path)

        captured = {}

        def fake_trainer(triplets, output_dir, base_model, **kwargs):
            captured["count"] = len(triplets)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "model.safetensors").write_text("fake", encoding="utf-8")
            return {"mrr": 0.99, "recall_at_5": 1.0, "recall_at_10": 1.0, "ndcg_at_10": 0.99, "num_triplets": len(triplets)}

        metrics = train_global(
            "train1",
            [_triplet("q", "p", "n")],
            trainer_fn=fake_trainer,
            settings=settings,
        )
        assert captured["count"] == 1
        assert metrics["mrr"] == 0.99
        # training_metrics.json written by the real trainer wrapper.
        assert (Path(settings.lecture_dir("train1")) / "training_metrics.json").exists()

    def test_train_global_rejects_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError):
            train_global("train1", [], settings=CloudSettings())


class TestInstall:
    def test_install_swaps_model_into_global_dir(self, tmp_path):
        train_dir = tmp_path / "train"
        (train_dir / "reranker_model").mkdir(parents=True)
        (train_dir / "reranker_model" / "model.safetensors").write_text("weights", encoding="utf-8")

        global_dir = tmp_path / "models" / "global_reranker"
        # Pre-existing stale model must be replaced atomically.
        global_dir.mkdir(parents=True)
        (global_dir / "stale.txt").write_text("old", encoding="utf-8")

        copied = install_global_model(train_dir, global_dir)
        assert copied == 1
        assert (global_dir / "model.safetensors").exists()
        assert not (global_dir / "stale.txt").exists()

    def test_install_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            install_global_model(tmp_path / "train", tmp_path / "models" / "g")
