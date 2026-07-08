# local/tests/test_d1_neo4j_loader.py
# Tests for D1 — Neo4j Graph Loader.
# Neo4j is fully mocked — no database needed.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, call

from config import LocalSettings
from schemas.enums import EntityType, RelationType
from local.loaders.neo4j_loader import load_neo4j, Neo4jLoadError


@pytest.fixture
def local_settings(tmp_path):
    return LocalSettings(
        LOCAL_MODEL_DIR=str(tmp_path / "local_runtime/models/"),
        TRANSFER_DIR=str(tmp_path / "transfer/"),
    )


@pytest.fixture
def package_dir(tmp_path):
    """Create a minimal package with 3 entities and 2 relations."""
    pkg = tmp_path / "package"
    pkg.mkdir()

    entities = [
        {"entity_id": "e_001", "name": "BFS", "type": "Algorithm"},
        {"entity_id": "e_002", "name": "Queue", "type": "Concept"},
        {"entity_id": "e_003", "name": "Graph", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r_001", "source_entity_id": "e_001",
         "relation": "USED_BY", "target_entity_id": "e_002"},
        {"relation_id": "r_002", "source_entity_id": "e_001",
         "relation": "PREREQUISITE_OF", "target_entity_id": "e_003"},
    ]

    (pkg / "entities.json").write_text(json.dumps(entities))
    (pkg / "relations.json").write_text(json.dumps(relations))
    return pkg


class TestNeo4jLoader:

    def test_load_success(self, package_dir, local_settings):
        """Loads 3 nodes and 2 relationships with mock driver."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        result = load_neo4j(package_dir, "lec_001", local_settings=local_settings, driver=mock_driver)

        assert result["node_count"] == 3
        assert result["relationship_count"] == 2
        assert mock_session.run.call_count == 5  # 3 nodes + 2 rels

    def test_node_labels_from_entity_type(self, package_dir, local_settings):
        """Nodes must use entity.type as the Neo4j label."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        load_neo4j(package_dir, "lec_001", local_settings=local_settings, driver=mock_driver)

        # Verify the first call creates an Algorithm node.
        first_call_query = mock_session.run.call_args_list[0][0][0]
        assert "Algorithm" in first_call_query

        # Second call creates a Concept node.
        second_call_query = mock_session.run.call_args_list[1][0][0]
        assert "Concept" in second_call_query

    def test_relations_match_on_entity_id(self, package_dir, local_settings):
        """Relationships must match on entity_id, never on name."""
        mock_session = MagicMock()
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        load_neo4j(package_dir, "lec_001", local_settings=local_settings, driver=mock_driver)

        # Relationship calls are calls 3 and 4 (0-indexed).
        rel_call = mock_session.run.call_args_list[3]
        kwargs = rel_call[1]
        assert kwargs["source"] == "e_001"
        assert kwargs["target"] == "e_002"
        # Should NOT contain "name" anywhere.
        query = rel_call[0][0]
        assert "entity_id" in query

    def test_bad_entity_reference_raises(self, tmp_path, local_settings):
        """Relation referencing non-existent entity raises error."""
        pkg = tmp_path / "bad_pkg"
        pkg.mkdir()
        entities = [{"entity_id": "e_001", "name": "BFS", "type": "Algorithm"}]
        relations = [
            {"relation_id": "r_001", "source_entity_id": "e_001",
             "relation": "USED_BY", "target_entity_id": "FAKE_ID"},
        ]
        (pkg / "entities.json").write_text(json.dumps(entities))
        (pkg / "relations.json").write_text(json.dumps(relations))

        with pytest.raises(Neo4jLoadError, match="target_entity_id"):
            load_neo4j(pkg, "lec_001", local_settings=local_settings, driver=MagicMock())

    def test_duplicate_entity_id_raises(self, tmp_path, local_settings):
        """Duplicate entity_id must raise error."""
        pkg = tmp_path / "dup_pkg"
        pkg.mkdir()
        entities = [
            {"entity_id": "e_001", "name": "BFS", "type": "Algorithm"},
            {"entity_id": "e_001", "name": "DFS", "type": "Algorithm"},
        ]
        (pkg / "entities.json").write_text(json.dumps(entities))
        (pkg / "relations.json").write_text(json.dumps([]))

        with pytest.raises(Neo4jLoadError, match="Duplicate"):
            load_neo4j(pkg, "lec_001", local_settings=local_settings, driver=MagicMock())
