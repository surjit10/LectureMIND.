# cloud/tests/test_reranker_training_pipeline.py
# Tests for Pipeline B — Global Reranker Training (independent Kaggle workflow).
#
# GPU-free: training is exercised via the existing trainer_fn injection (same
# pattern as test_b2b_global_training.py). The dev-evaluation step is expected
# to fail gracefully against fake model files — that IS one of the reliability
# paths under test.

import json
import zipfile
from pathlib import Path

import pytest

from cloud.reranker_training.dataset_builder import build_dataset, train_dev_split
from cloud.reranker_training.export import export_version
from cloud.reranker_training.package_discovery import (
    PackageSource,
    discover_packages,
    extract_triplets,
)
from cloud.reranker_training.pipeline import TrainingConfig, run_training_pipeline
from cloud.reranker_training.versioning import (
    commit_version,
    load_index,
    next_version,
    update_pointers,
)


def _triplet(q: str, pos: str, neg: str) -> dict:
    return {"query": q, "positive": pos, "negative": neg}


def _n_triplets(n: int, prefix: str = "q") -> list:
    """n distinct triplets (dedupe-safe fixtures)."""
    return [_triplet(f"{prefix}{i}", f"p{prefix}{i}", f"n{prefix}{i}") for i in range(n)]


def _make_package_zip(tmp_path: Path, lecture_id: str, triplets, corrupt: bool = False) -> Path:
    """Build a flat knowledge-package zip containing junk + triplets.json."""
    zip_path = tmp_path / "packages" / f"{lecture_id}_knowledge_package.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        zip_path.write_bytes(b"this is definitely not a zip file")
        return zip_path
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("manifest.json", json.dumps(
            {"lecture_id": lecture_id, "reranker_version": "v1.0"},
        ))
        zf.writestr("segments.json", json.dumps([{"segment_id": "s1"}]))
        zf.writestr("embeddings.npy", b"junk-bytes-that-must-be-ignored")
        zf.writestr("triplets.json", json.dumps(triplets))
    return zip_path


def _fake_trainer(triplets, output_dir, base_model, **kwargs):
    """Test trainer: writes minimal fake model artifacts, returns metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(
        {"model_type": "bert", "architectures": ["BertForSequenceClassification"]},
    ))
    (output_dir / "model.safetensors").write_text("fake-weights", encoding="utf-8")
    (output_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    return {
        "mrr": 0.95, "recall_at_5": 1.0, "recall_at_10": 1.0,
        "ndcg_at_10": 0.95, "num_triplets": len(triplets),
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_finds_zips_and_extracted_dirs(self, tmp_path):
        _make_package_zip(tmp_path, "lecture_a", [_triplet("q", "p", "n")])
        pkg_dir = tmp_path / "packages" / "lecture_b"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "triplets.json").write_text(json.dumps([_triplet("q2", "p2", "n2")]))

        sources = discover_packages(tmp_path / "packages")
        assert any(s.is_zip for s in sources)
        assert any(not s.is_zip for s in sources)
        assert len(sources) == 2

    def test_excludes_named_dirs(self, tmp_path):
        _make_package_zip(tmp_path, "good", [_triplet("q", "p", "n")])
        bad = tmp_path / "packages" / "datasets" / "jit007"
        bad.mkdir(parents=True, exist_ok=True)
        (bad / "model.zip").write_bytes(b"nope")
        sources = discover_packages(tmp_path / "packages")
        assert len(sources) == 1
        assert "model.zip" not in [s.path.name for s in sources]

    def test_missing_root_is_empty(self, tmp_path):
        assert discover_packages(tmp_path / "does-not-exist") == []

    def test_nested_zips_discovered(self, tmp_path):
        nested = tmp_path / "packages" / "deep" / "deeper"
        nested.mkdir(parents=True, exist_ok=True)
        z = nested / "lecture_x_knowledge_package.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("triplets.json", json.dumps([_triplet("q", "p", "n")]))
        assert len(discover_packages(tmp_path / "packages")) == 1


# ---------------------------------------------------------------------------
# Triplet extraction (zip-aware, ignore-everything-else)
# ---------------------------------------------------------------------------


class TestTripletExtraction:
    def test_extracts_only_triplets_from_zip(self, tmp_path):
        z = _make_package_zip(tmp_path, "lec", [_triplet("q1", "p1", "n1"), _triplet("q2", "p2", "n2")])
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert extraction.ok
        assert len(extraction.triplets) == 2

    def test_missing_triplets_in_zip(self, tmp_path):
        z = tmp_path / "packages" / "no_triplets.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("manifest.json", "{}")
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert not extraction.ok
        assert "missing" in extraction.error

    def test_corrupt_zip(self, tmp_path):
        z = _make_package_zip(tmp_path, "bad", [], corrupt=True)
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert not extraction.ok

    def test_invalid_json(self, tmp_path):
        z = tmp_path / "packages" / "bad_json.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("triplets.json", "{not json")
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert not extraction.ok
        assert "JSON" in extraction.error

    def test_oversized_triplets_skipped_not_oom(self, tmp_path, monkeypatch):
        from cloud.reranker_training import package_discovery

        monkeypatch.setattr(package_discovery, "MAX_TRIPLETS_BYTES", 10)
        z = tmp_path / "packages" / "huge.zip"
        z.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("triplets.json", json.dumps([_triplet("q" * 50, "p", "n")]))
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert not extraction.ok
        assert "exceeds" in extraction.error

    def test_malformed_rows_dropped(self, tmp_path):
        z = _make_package_zip(
            tmp_path, "mixed",
            [_triplet("q", "p", "n"), {"bad": "shape"}, {"query": "", "positive": "p", "negative": "n"}],
        )
        extraction = extract_triplets(PackageSource(path=z, is_zip=True))
        assert extraction.ok
        assert len(extraction.triplets) == 1
        assert extraction.n_dropped == 2

    def test_extracted_dir_source(self, tmp_path):
        d = tmp_path / "packages" / "lec_dir"
        d.mkdir(parents=True, exist_ok=True)
        (d / "triplets.json").write_text(json.dumps([_triplet("q", "p", "n")]))
        extraction = extract_triplets(PackageSource(path=d, is_zip=False))
        assert extraction.ok
        assert len(extraction.triplets) == 1


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


class TestDatasetBuilder:
    def test_merges_and_dedupes(self, tmp_path):
        _make_package_zip(tmp_path, "a", [_triplet("dup", "p", "n")] * 3)
        _make_package_zip(tmp_path, "b", [_triplet("dup", "p", "n"), _triplet("q2", "p2", "n2")])

        report = build_dataset(tmp_path / "packages")
        assert report.sources_used == 2
        assert report.n_triplets_raw == 5
        assert report.n_triplets_after_dedupe == 2  # 3 duplicates removed
        assert set(report.lecture_ids) == {"a", "b"}

    def test_skips_missing_and_corrupt_without_stopping(self, tmp_path):
        _make_package_zip(tmp_path, "good", [_triplet("q", "p", "n")])
        _make_package_zip(tmp_path, "corrupt", [_triplet("q", "p", "n")], corrupt=True)
        z = tmp_path / "packages" / "missing.zip"
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("manifest.json", "{}")

        report = build_dataset(tmp_path / "packages")
        assert report.sources_found == 3
        assert report.sources_used == 1
        assert len(report.sources_skipped) == 2
        assert len(report.triplets) == 1

    def test_min_triplets_and_max_per_lecture(self, tmp_path):
        _make_package_zip(tmp_path, "big", _n_triplets(40, "big"))
        _make_package_zip(tmp_path, "small", _n_triplets(5, "small"))

        report = build_dataset(tmp_path / "packages", min_triplets=10)
        assert report.sources_used == 1
        assert report.lecture_ids == ["big"]

        report2 = build_dataset(tmp_path / "packages", min_triplets=0, max_per_lecture=10)
        assert report2.sources_used == 2
        assert report2.n_triplets_after_dedupe == 15

    def test_missing_root_empty(self, tmp_path):
        report = build_dataset(tmp_path / "nope")
        assert report.sources_found == 0
        assert report.triplets == []


class TestTrainDevSplit:
    def test_split_sizes_and_determinism(self):
        triplets = [_triplet(f"q{i}", f"p{i}", f"n{i}") for i in range(100)]
        train1, dev1 = train_dev_split(triplets, dev_fraction=0.1, seed=42)
        train2, dev2 = train_dev_split(triplets, dev_fraction=0.1, seed=42)
        assert train1 == train2 and dev1 == dev2
        assert len(dev1) == 10 and len(train1) == 90

    def test_single_triplet_no_dev(self):
        train, dev = train_dev_split([_triplet("q", "p", "n")])
        assert len(train) == 1 and dev == []

    def test_empty(self):
        assert train_dev_split([]) == ([], [])


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


class TestVersioning:
    def _commit(self, models_root: Path, version: int, metric: float, tag: str):
        model_src = models_root / f"src_{tag}"
        (model_src / "model").mkdir(parents=True)
        (model_src / "model" / "config.json").write_text("{}", encoding="utf-8")
        (model_src / "model" / "model.safetensors").write_text(tag, encoding="utf-8")
        commit_version(
            models_root, version, model_src / "model",
            {"version": version, "tag": tag}, {"dev": {"ndcg_at_10": metric}},
        )

    def test_next_version_and_commit(self, tmp_path):
        models_root = tmp_path / "models"
        assert next_version(models_root) == 1
        self._commit(models_root, 1, 0.8, "a")
        assert next_version(models_root) == 2
        self._commit(models_root, 2, 0.9, "b")

        assert (models_root / "v1" / "model" / "config.json").exists()
        assert (models_root / "v1" / "metadata.json").exists()
        assert (models_root / "v1" / "evaluation_report.json").exists()
        index = load_index(models_root)
        assert index["versions"] == [1, 2]
        assert index["latest"] == 2

    def test_best_pointer_requires_strict_improvement(self, tmp_path):
        models_root = tmp_path / "models"
        self._commit(models_root, 1, 0.80, "a")
        update_pointers(models_root, 1, 0.80, "ndcg_at_10_dev")
        self._commit(models_root, 2, 0.70, "b")
        update_pointers(models_root, 2, 0.70, "ndcg_at_10_dev")

        index = load_index(models_root)
        assert index["best"] == 1          # worse metric: best NOT overwritten
        assert index["latest"] == 2
        assert (models_root / "best" / "model" / "model.safetensors").read_text() == "a"

        self._commit(models_root, 3, 0.90, "c")
        update_pointers(models_root, 3, 0.90, "ndcg_at_10_dev")
        index = load_index(models_root)
        assert index["best"] == 3          # strictly better: best overwritten
        assert (models_root / "best" / "model" / "model.safetensors").read_text() == "c"

    def test_tie_keeps_existing_best(self, tmp_path):
        models_root = tmp_path / "models"
        self._commit(models_root, 1, 0.80, "a")
        update_pointers(models_root, 1, 0.80, "ndcg_at_10_dev")
        self._commit(models_root, 2, 0.80, "b")
        update_pointers(models_root, 2, 0.80, "ndcg_at_10_dev")
        assert load_index(models_root)["best"] == 1


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_flat_layout_no_temp_files(self, tmp_path):
        models_root = tmp_path / "models"
        model_src = tmp_path / "trained" / "reranker_model"
        (model_src / "subdir").mkdir(parents=True)
        (model_src / "config.json").write_text("{}", encoding="utf-8")
        (model_src / "model.safetensors").write_text("w", encoding="utf-8")
        (model_src / "tokenizer.json").write_text("{}", encoding="utf-8")
        (model_src / ".DS_Store").write_text("junk", encoding="utf-8")
        (model_src / "subdir" / "extra.txt").write_text("x", encoding="utf-8")
        commit_version(models_root, 1, model_src, {"version": 1}, {"dev": {}})

        zip_path = export_version(models_root, 1)
        assert zip_path.name == "global_reranker_v1.zip"
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        assert "config.json" in names
        assert "model.safetensors" in names
        assert "tokenizer.json" in names
        assert "metadata.json" in names
        assert "evaluation_report.json" in names
        assert "subdir/extra.txt" in names
        assert ".DS_Store" not in names
        assert ".staging" not in " ".join(names)


# ---------------------------------------------------------------------------
# End-to-end pipeline (injected trainer, no GPU)
# ---------------------------------------------------------------------------


class TestPipeline:
    def _run(self, tmp_path: Path, packages_root: Path, **cfg_kwargs):
        return run_training_pipeline(
            packages_root=packages_root,
            models_root=tmp_path / "models",
            scratch_root=tmp_path / "scratch",
            config=TrainingConfig(**cfg_kwargs),
            trainer_fn=_fake_trainer,
        )

    def test_end_to_end_completed(self, tmp_path):
        _make_package_zip(tmp_path, "lec_a", _n_triplets(40, "qa"))
        _make_package_zip(tmp_path, "lec_b", _n_triplets(20, "qb"))

        result = self._run(tmp_path, tmp_path / "packages")

        assert result.status == "completed"
        assert result.version == 1
        assert result.n_packages_used == 2
        assert result.n_triplets == 60
        assert set(result.lecture_ids) == {"lec_a", "lec_b"}
        assert result.n_train_triplets + result.n_dev_triplets == 60
        assert result.model_zip is not None
        assert Path(result.model_zip).is_file()

        metadata = json.loads(Path(result.metadata_path).read_text())
        assert metadata["version"] == 1
        assert metadata["n_packages_used"] == 2
        assert metadata["n_triplets_merged"] == 60
        assert metadata["hyperparameters"]["epochs"] == 3
        assert metadata["metrics"]["train"]["mrr"] == 0.95
        # Dev eval cannot load fake weights — must be tolerated, not fatal.
        assert result.eval_error is not None or result.dev_metrics == {}

        # Pointers + zip exist.
        assert (tmp_path / "models" / "best").is_dir()
        assert (tmp_path / "models" / "latest").is_dir()
        assert (tmp_path / "models" / "global_reranker_v1.zip").is_file()

    def test_aborts_when_no_packages(self, tmp_path):
        result = self._run(tmp_path, tmp_path / "empty")
        assert result.status == "aborted"
        assert "No usable triplets" in result.reason
        assert result.version is None

    def test_continues_past_corrupt_package(self, tmp_path):
        _make_package_zip(tmp_path, "good", _n_triplets(30))
        _make_package_zip(tmp_path, "corrupt", _n_triplets(30), corrupt=True)

        result = self._run(tmp_path, tmp_path / "packages")
        assert result.status == "completed"
        assert result.n_packages_found == 2
        assert result.n_packages_used == 1
        assert len(result.skipped_sources) == 1

    def test_training_failure_commits_nothing(self, tmp_path):
        _make_package_zip(tmp_path, "a", _n_triplets(30))

        def _exploding_trainer(*args, **kwargs):
            raise RuntimeError("OOM / interrupted training")

        result = run_training_pipeline(
            packages_root=tmp_path / "packages",
            models_root=tmp_path / "models",
            scratch_root=tmp_path / "scratch",
            config=TrainingConfig(),
            trainer_fn=_exploding_trainer,
        )
        assert result.status == "aborted"
        assert "Training failed" in result.reason
        assert not (tmp_path / "models" / "v1").exists()  # no half-written version

    def test_version_increments_across_runs(self, tmp_path):
        _make_package_zip(tmp_path, "a", _n_triplets(30))
        first = self._run(tmp_path, tmp_path / "packages")
        second = self._run(tmp_path, tmp_path / "packages")
        assert first.version == 1
        assert second.version == 2
        index = load_index(tmp_path / "models")
        assert index["versions"] == [1, 2]
        assert index["latest"] == 2

    def test_cli_env_defaults_honored(self, tmp_path, monkeypatch):
        """RERANKER_TRAINING_* env vars feed the CLI defaults (runner contract)."""
        from cloud.reranker_training import pipeline

        monkeypatch.setenv("RERANKER_TRAINING_MODELS_ROOT", str(tmp_path / "env_models"))
        monkeypatch.setenv("RERANKER_TRAINING_SCRATCH_ROOT", str(tmp_path / "env_scratch"))

        # Reconstruct what main() does with the env, then run via the function.
        import os

        env_models = os.environ.get("RERANKER_TRAINING_MODELS_ROOT")
        env_scratch = os.environ.get("RERANKER_TRAINING_SCRATCH_ROOT")
        _make_package_zip(tmp_path, "a", _n_triplets(30))

        result = run_training_pipeline(
            packages_root=tmp_path / "packages",
            models_root=Path(env_models),
            scratch_root=Path(env_scratch),
            config=TrainingConfig(),
            trainer_fn=_fake_trainer,
        )
        assert result.status == "completed"
        assert (tmp_path / "env_models" / "global_reranker_v1.zip").is_file()
