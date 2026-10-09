"""Single authenticated Neo4j driver factory for local runtime code."""

from __future__ import annotations

from typing import Any

from config import LocalSettings


def create_neo4j_driver(settings: LocalSettings | None = None) -> Any:
    settings = settings or LocalSettings()
    from neo4j import GraphDatabase

    user = settings.NEO4J_USER.strip()
    password = settings.NEO4J_PASSWORD
    if bool(user) != bool(password):
        raise ValueError("NEO4J_USER and NEO4J_PASSWORD must be configured together.")
    if user:
        return GraphDatabase.driver(settings.NEO4J_URI, auth=(user, password))
    return GraphDatabase.driver(settings.NEO4J_URI)
