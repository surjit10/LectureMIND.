from evaluation.load_testing import load_test


def test_reported_concurrency_is_real_worker_count(monkeypatch, tmp_path):
    monkeypatch.setattr(
        load_test,
        "_send_query",
        lambda api_url, query: {
            "ttft": 0.01, "latency": 0.02, "status": 200, "error": None,
        },
    )
    result = load_test.run_load_test(
        user_levels=[3], requests_per_level=7, report_dir=tmp_path
    )[0]
    assert result["concurrency"] == 3
    assert result["total_requests"] == 7
    assert result["successful"] == 7
    assert result["error_rate"] == 0.0
    assert result["p50_latency"] == 0.02
    assert result["p95_latency"] == 0.02
    assert result["p99_latency"] == 0.02


def test_invalid_concurrency_rejected(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="positive"):
        load_test.run_load_test(user_levels=[0], report_dir=tmp_path)
