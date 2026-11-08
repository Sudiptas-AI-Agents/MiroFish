"""Real Neo4j + LiteLLM round trip. Run with RUN_GRAPHITI_INTEGRATION=1."""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import pytest
from pydantic import Field
from zep_cloud import EntityEdgeSourceTarget, NotFoundError
from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText

from app.config import Config
from app.services.graph_builder import GraphBuilderService
from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GRAPHITI_INTEGRATION") != "1",
    reason="needs the real db.lan Neo4j and LiteLLM",
)

CHUNK = (
    "Alice Chen works for Acme Robotics in Berlin. Bob Singh also works for Acme "
    "Robotics. Alice Chen manages Bob Singh."
)


class Person(EntityModel):
    """A human being mentioned in the text."""

    role: Optional[EntityText] = Field(description="the person's job", default=None)


class Company(EntityModel):
    """A company or organisation."""


class WorksFor(EdgeModel):
    """A person is employed by a company."""


def test_ingest_read_search_delete(monkeypatch):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    from app.utils.graphiti_backend import get_graphiti_backend

    backend = get_graphiti_backend()
    graph_id = f"mirofish_test_{uuid.uuid4().hex[:12]}"
    builder = object.__new__(GraphBuilderService)
    builder.client = backend

    backend.graph.create(graph_id=graph_id, name="integration", description="throwaway")
    try:
        backend.graph.set_ontology(
            graph_ids=[graph_id],
            entities={"Person": Person, "Company": Company},
            edges={
                "WORKS_FOR": (
                    WorksFor,
                    [EntityEdgeSourceTarget(source="Person", target="Company")],
                )
            },
        )
        submission = builder.add_text_batches(graph_id, [CHUNK], batch_size=1)
        builder._wait_for_batch(submission, timeout=600)

        nodes = fetch_all_nodes(backend, graph_id)
        edges = fetch_all_edges(backend, graph_id)
        assert any("Alice" in node.name for node in nodes)
        assert any("Person" in node.labels for node in nodes)
        assert edges

        found = backend.graph.search(
            graph_id=graph_id, query="Who works for Acme?", limit=5, scope="edges", reranker="rrf"
        )
        assert found.edges

        assert backend.graph.node.get(uuid_=nodes[0].uuid_).name == nodes[0].name
    finally:
        backend.graph.delete(graph_id=graph_id)

    assert fetch_all_nodes(backend, graph_id) == []
    with pytest.raises(NotFoundError):
        backend.graph.get(graph_id)


FIXTURES = Path(__file__).parent / "fixtures" / "graphiti_replay"


def test_real_simulation_batches_all_ingest(monkeypatch):
    """Replays the activity batches that failed in the first end-to-end run."""

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    from app.utils.graphiti_backend import get_graphiti_backend

    backend = get_graphiti_backend()
    graph_id = f"mirofish_test_{uuid.uuid4().hex[:12]}"
    ontology = json.loads((FIXTURES / "ontology.json").read_text(encoding="utf-8"))
    batches = json.loads((FIXTURES / "batches.json").read_text(encoding="utf-8"))

    def ingest(data, created_at=None):
        episode = backend.graph.add(
            graph_id=graph_id, type="text", data=data, created_at=created_at,
            source_description="MiroFish simulation activity batch",
        )
        deadline = time.time() + 900
        while not backend.graph.episode.get(uuid_=episode.uuid_).processed:
            assert time.time() < deadline, "episode timed out"
            time.sleep(2)

    backend.graph.create(graph_id=graph_id, name="replay", description="throwaway")
    try:
        backend._run(backend._store.save_ontology(graph_id, ontology))
        ingest((FIXTURES / "source.txt").read_text(encoding="utf-8"))
        for batch in batches:
            ingest(batch["data"], batch["created_at"])
        assert fetch_all_nodes(backend, graph_id)
    finally:
        backend.graph.delete(graph_id=graph_id)
