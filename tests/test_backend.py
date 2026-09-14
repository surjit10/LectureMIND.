# SMOKE SCRIPT (not a pytest test): manual live-server check; excluded from pytest via pytest.ini addopts.
import logging
import json
import sys

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("test_pipeline")

from local.llm.provider_registry import get_provider_registry

reg = get_provider_registry()
logger.info("Current config: %s", json.dumps(reg.to_dict(), indent=2))

backend = reg.get_active_backend()
logger.info("Backend class: %s", type(backend).__name__)
