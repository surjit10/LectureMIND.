from unittest.mock import patch

import pytest

from config import LocalSettings
from local.storage.neo4j_connection import create_neo4j_driver


def test_neo4j_driver_without_auth_for_local_dev():
    with patch("neo4j.GraphDatabase.driver") as driver:
        create_neo4j_driver(LocalSettings(NEO4J_USER="", NEO4J_PASSWORD=""))
        driver.assert_called_once_with("bolt://localhost:7687")


def test_neo4j_driver_uses_configured_credentials():
    with patch("neo4j.GraphDatabase.driver") as driver:
        create_neo4j_driver(LocalSettings(NEO4J_USER="neo4j", NEO4J_PASSWORD="secret"))
        driver.assert_called_once_with(
            "bolt://localhost:7687", auth=("neo4j", "secret")
        )


def test_partial_neo4j_credentials_are_rejected():
    with pytest.raises(ValueError, match="configured together"):
        create_neo4j_driver(LocalSettings(NEO4J_USER="neo4j", NEO4J_PASSWORD=""))
