# tests/conftest.py
# test_flow.py and test_set_active.py are manual smoke-test scripts that
# hit a running server at http://localhost:8000 — they are not unit tests
# and must not be collected by pytest (which errored at import/collection).
collect_ignore = ["test_flow.py", "test_set_active.py"]
