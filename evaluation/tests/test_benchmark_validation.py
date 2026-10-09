import json

from evaluation.validate_benchmark import validate_benchmark


def _sample(index: int, **overrides):
    sample = {
        "lecture_id": f"lecture_{index % 2}",
        "query": f"question {index}",
        "ground_truth_answer": f"answer {index}",
        "expected_chunk_ids": [f"chunk_{index}"],
        "expected_route": "vector_only" if index % 2 else "graph_only",
        "question_type": "visual" if index % 10 == 0 else "factual",
        "difficulty": "hard" if index % 10 == 0 else "medium",
        "topic": "topic",
        "need_visual": index % 10 == 0,
    }
    sample.update(overrides)
    return sample


def test_smoke_profile_checks_structure(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([_sample(1)]))
    report = validate_benchmark(path, "smoke")
    assert report["status"] == "PASS"
    assert report["warnings"]


def test_release_profile_requires_diversity(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([_sample(1, lecture_id="only", expected_route="vector_only")]))
    report = validate_benchmark(path, "release")
    assert report["status"] == "FAIL"
    assert any("at least 2 lectures" in issue for issue in report["issues"])
    assert any("at least 2 expected routes" in issue for issue in report["issues"])


def test_release_profile_accepts_balanced_dataset(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([_sample(i) for i in range(100)]))
    report = validate_benchmark(path, "release")
    assert report["status"] == "PASS", report["issues"]


def test_duplicate_queries_fail_validation(tmp_path):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([_sample(1), _sample(2, query=" QUESTION   1 ")]))
    report = validate_benchmark(path, "smoke")
    assert report["status"] == "FAIL"
    assert any("duplicate normalized queries" in issue for issue in report["issues"])
