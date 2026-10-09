from mlops.run_manifest import create_run_manifest


def test_run_manifest_captures_reproducibility_metadata():
    manifest = create_run_manifest(
        "unit_test", parameters={"seed": 42}, dependency_names=("pydantic",)
    )
    assert manifest["run_type"] == "unit_test"
    assert manifest["run_id"].startswith("unit_test_")
    assert manifest["parameters"] == {"seed": 42}
    assert "git_commit" in manifest["code"]
    assert manifest["environment"]["python"]
    assert "pydantic" in manifest["environment"]["dependencies"]
