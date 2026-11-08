"""GraphBackend contract: the Zep SDK and the Graphiti backend expose one surface."""

import asyncio
import inspect
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional

import pytest
from neo4j.exceptions import ServiceUnavailable
from pydantic import Field
from zep_cloud import BatchAddItem, EntityEdgeSourceTarget, NotFoundError
from zep_cloud.client import Zep
from zep_cloud.external_clients.ontology import EdgeModel, EntityModel, EntityText

from app.config import Config
from app.services.graph_builder import GraphBuilderService
from app.utils import zep
from app.utils.zep_paging import fetch_all_edges, fetch_all_nodes

graphiti_backend = pytest.importorskip("app.utils.graphiti_backend")
GRAPH = "mirofish_contract"

# Every call shape the application makes, copied from its call sites.
SURFACE = [
    ("graph.create", {"graph_id": GRAPH, "name": "n", "description": "d"}),
    ("graph.get", {"graph_id": GRAPH}),
    ("graph.delete", {"graph_id": GRAPH}),
    ("graph.set_ontology", {"graph_ids": [GRAPH], "entities": {}, "edges": None}),
    (
        "graph.add",
        {
            "graph_id": GRAPH,
            "type": "text",
            "data": "x",
            "created_at": "2026-01-01T00:00:00+00:00",
            "source_description": "s",
            "metadata": {"a": 1},
        },
    ),
    (
        "graph.search",
        {"graph_id": GRAPH, "query": "q", "limit": 5, "scope": "edges", "reranker": "rrf"},
    ),
    ("graph.node.get", {"uuid_": "u"}),
    ("graph.node.get_edges", {"node_uuid": "u"}),
    ("graph.node.with_raw_response.get_by_graph_id", {"graph_id": GRAPH, "limit": 1, "cursor": "c"}),
    ("graph.edge.with_raw_response.get_by_graph_id", {"graph_id": GRAPH, "limit": 1, "cursor": "c"}),
    ("graph.episode.get", {"uuid_": "u"}),
    ("batch.create", {"metadata": {"a": 1}}),
    ("batch.add", {"batch_id": "b", "items": []}),
    ("batch.process", {"batch_id": "b"}),
    ("batch.get", {"batch_id": "b"}),
    ("batch.list", {"limit": 100, "cursor": 1}),
    ("batch.list_items", {"batch_id": "b", "limit": 100, "cursor": 1}),
]


class FakeStore:
    """In-memory stand-in for the Neo4j/Graphiti gateway."""

    def __init__(self, fail_on=()):
        self.graphs = {}
        self.episodes = []
        self.searches = []
        self.nodes = []
        self.edges = []
        self.fail_on = set(fail_on)
        self.in_flight = 0
        self.max_in_flight = 0

    async def save_graph(self, graph_id, name, description):
        self.graphs[graph_id] = {"name": name, "description": description, "ontology": None}

    async def load_graph(self, graph_id):
        return self.graphs.get(graph_id)

    async def save_ontology(self, graph_id, ontology):
        self.graphs[graph_id]["ontology"] = ontology

    async def delete_graph(self, graph_id):
        self.graphs.pop(graph_id)

    async def add_episode(self, graph_id, name, body, source_description, reference_time, ontology):
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.005)
        self.in_flight -= 1
        if body in self.fail_on:
            raise RuntimeError(f"extraction failed for {body}")
        self.episodes.append((graph_id, body, reference_time, ontology))

    async def search(self, graph_id, query, limit, scope):
        self.searches.append((graph_id, query, limit, scope))
        return self.edges[:limit], self.nodes[:limit]

    async def nodes_page(self, graph_id, limit, cursor):
        return _page(self.nodes, limit, cursor)

    async def edges_page(self, graph_id, limit, cursor):
        return _page(self.edges, limit, cursor)

    async def get_node(self, node_uuid):
        return next((n for n in self.nodes if n.uuid == node_uuid), None)

    async def node_edges(self, node_uuid):
        return [e for e in self.edges if node_uuid in (e.source_node_uuid, e.target_node_uuid)]


def _page(items, limit, cursor):
    ordered = sorted(items, key=lambda item: item.uuid, reverse=True)
    if cursor:
        ordered = [item for item in ordered if item.uuid < cursor]
    return ordered[:limit]


def _node(uuid):
    return SimpleNamespace(
        uuid=uuid, name=f"n{uuid}", labels=["Entity", "Person"], summary="s",
        attributes={"role": "r"}, created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _edge(uuid, source="1", target="2"):
    return SimpleNamespace(
        uuid=uuid, name="KNOWS", fact="f", source_node_uuid=source, target_node_uuid=target,
        attributes={}, created_at=None, valid_at=None, invalid_at=None, expired_at=None,
        episodes=["e1"],
    )


@pytest.fixture
def store():
    return FakeStore()


@pytest.fixture
def backend(store):
    return graphiti_backend.GraphitiBackend(store)


def _resolve(root, path):
    target = root
    for part in path.split("."):
        target = getattr(target, part)
    return target


@pytest.mark.parametrize("path,kwargs", SURFACE, ids=[entry[0] for entry in SURFACE])
@pytest.mark.parametrize("backend_name", ["zep", "graphiti"])
def test_both_backends_accept_every_call_the_app_makes(backend_name, path, kwargs, backend):
    root = Zep(api_key="unused") if backend_name == "zep" else backend
    inspect.signature(_resolve(root, path)).bind(**kwargs)


def test_rejects_graph_ids_outside_the_mirofish_namespace(backend):
    with pytest.raises(ValueError, match="mirofish_"):
        backend.graph.create(graph_id="someone_else", name="n", description="d")
    with pytest.raises(ValueError, match="mirofish_"):
        backend.graph.delete(graph_id="someone_else")


def test_get_and_delete_of_a_missing_graph_raise_not_found(backend):
    with pytest.raises(NotFoundError):
        backend.graph.get(GRAPH)
    with pytest.raises(NotFoundError):
        backend.graph.delete(graph_id=GRAPH)


def test_created_graph_can_be_read_and_deleted(backend, store):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    assert backend.graph.get(GRAPH).graph_id == GRAPH
    backend.graph.delete(graph_id=GRAPH)
    assert GRAPH not in store.graphs


def _wait_processed(backend, episode_uuid):
    deadline = time.time() + 5
    while time.time() < deadline:
        episode = backend.graph.episode.get(uuid_=episode_uuid)
        if episode.processed:
            return episode
        time.sleep(0.01)
    raise AssertionError("episode never processed")


def test_add_is_asynchronous_and_passes_the_reference_time(backend, store):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    episode = backend.graph.add(
        graph_id=GRAPH, type="text", data="hello",
        created_at="2026-03-01T10:00:00+00:00", source_description="s",
    )
    _wait_processed(backend, episode.uuid_)
    assert store.episodes[0][:2] == (GRAPH, "hello")
    assert store.episodes[0][2] == datetime(2026, 3, 1, 10, tzinfo=timezone.utc)


def test_failed_episode_surfaces_when_polled():
    store = FakeStore(fail_on={"bad"})
    backend = graphiti_backend.GraphitiBackend(store)
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    episode = backend.graph.add(graph_id=GRAPH, type="text", data="bad", source_description="s")
    with pytest.raises(RuntimeError, match="extraction failed"):
        _wait_processed(backend, episode.uuid_)


def test_unknown_episode_is_not_found(backend):
    with pytest.raises(NotFoundError):
        backend.graph.episode.get(uuid_="never-issued")


def _builder(backend):
    builder = object.__new__(GraphBuilderService)
    builder.client = backend
    return builder


def test_builder_batch_flow_ingests_in_order_one_at_a_time_per_graph(backend, store):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    builder = _builder(backend)
    chunks = [f"chunk {i}" for i in range(7)]

    submission = builder.add_text_batches(GRAPH, chunks, batch_size=3)
    uuids = builder._wait_for_batch(submission, timeout=5)

    assert [episode[1] for episode in store.episodes] == chunks
    assert uuids == submission.episode_uuids and len(set(uuids)) == 7
    assert store.max_in_flight == 1


def test_builder_batch_flow_fails_when_one_chunk_fails():
    store = FakeStore(fail_on={"chunk 1"})
    backend = graphiti_backend.GraphitiBackend(store)
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    builder = _builder(backend)

    submission = builder.add_text_batches(GRAPH, ["chunk 0", "chunk 1", "chunk 2"])

    with pytest.raises(RuntimeError, match="extraction failed"):
        builder._wait_for_batch(submission, timeout=5)


def test_unknown_batch_after_restart_reads_as_failed_so_resume_is_refused(backend):
    assert backend.batch.get(batch_id="lost").status == "failed"


def test_batch_list_finds_a_batch_by_its_metadata(backend):
    backend.batch.create(metadata={"mirofish_operation_id": "op", "graph_id": GRAPH})
    builder = _builder(backend)
    assert builder._find_batch_by_operation_id(GRAPH, "op", max_attempts=1) is not None


def test_item_uses_the_zep_batch_item_type(backend, store):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    batch = backend.batch.create(metadata={})
    added = backend.batch.add(
        batch_id=batch.batch_id,
        items=[BatchAddItem(type="graph_episode", graph_id=GRAPH, data="d", data_type="text")],
    )
    assert added[0].sequence_index == 0 and added[0].episode_uuid


def test_paging_returns_every_node_and_edge(backend, store):
    store.nodes = [_node(str(i)) for i in range(5)]
    store.edges = [_edge(str(i)) for i in range(3)]

    nodes = fetch_all_nodes(backend, GRAPH, page_size=2)
    edges = fetch_all_edges(backend, GRAPH, page_size=100)

    assert sorted(n.uuid_ for n in nodes) == [str(i) for i in range(5)]
    assert nodes[0].labels == ["Entity", "Person"]
    assert len(edges) == 3 and edges[0].fact == "f" and edges[0].episodes == ["e1"]


def test_search_returns_zep_shaped_results(backend, store):
    store.nodes, store.edges = [_node("1")], [_edge("9")]

    result = backend.graph.search(
        graph_id=GRAPH, query="q", limit=4, scope="edges", reranker="cross_encoder"
    )

    assert store.searches == [(GRAPH, "q", 4, "edges")]
    assert result.edges[0].uuid_ == "9" and result.nodes[0].uuid_ == "1"


def test_search_rejects_an_unsupported_scope(backend):
    with pytest.raises(ValueError, match="scope"):
        backend.graph.search(graph_id=GRAPH, query="q", limit=1, scope="episodes")


def test_node_read_maps_missing_to_not_found_and_lists_edges(backend, store):
    store.nodes, store.edges = [_node("1")], [_edge("9", "1", "2")]
    assert backend.graph.node.get(uuid_="1").name == "n1"
    assert backend.graph.node.get_edges(node_uuid="2")[0].uuid_ == "9"
    with pytest.raises(NotFoundError):
        backend.graph.node.get(uuid_="absent")


class Person(EntityModel):
    """A person."""

    role: Optional[EntityText] = Field(description="their role", default=None)
    summary: Optional[EntityText] = Field(description="clashes with EntityNode", default=None)


class WorksFor(EdgeModel):
    """Employment."""

    since: Optional[str] = Field(description="start date", default=None)


def _set_ontology(backend):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    backend.graph.set_ontology(
        graph_ids=[GRAPH],
        entities={"Person": Person},
        edges={"WORKS_FOR": (WorksFor, [EntityEdgeSourceTarget(source="Person", target="Entity")])},
    )


def test_ontology_is_persisted_and_applied_to_every_episode(backend, store):
    _set_ontology(backend)
    episode = backend.graph.add(graph_id=GRAPH, type="text", data="x", source_description="s")
    _wait_processed(backend, episode.uuid_)
    assert store.episodes[0][3]["entities"]["Person"]["fields"]["role"] == "their role"


def test_ontology_maps_to_graphiti_pydantic_types(backend, store):
    _set_ontology(backend)

    types = graphiti_backend.ontology_to_graphiti(store.graphs[GRAPH]["ontology"])

    person = types["entity_types"]["Person"]
    assert person.__doc__ == "A person."
    assert person.model_fields["role"].description == "their role"
    assert "summary" not in person.model_fields and "entity_summary" in person.model_fields
    assert types["edge_types"]["WORKS_FOR"].model_fields["since"].description == "start date"
    assert types["edge_type_map"] == {("Person", "Entity"): ["WORKS_FOR"]}
    from graphiti_core.utils.ontology_utils.entity_types_utils import validate_entity_types
    assert validate_entity_types(types["entity_types"])


def test_ontology_rejects_a_type_name_unsafe_for_a_cypher_label(backend):
    backend.graph.create(graph_id=GRAPH, name="n", description="d")
    unsafe = type("Unsafe", (EntityModel,), {"__doc__": "x"})
    with pytest.raises(ValueError, match="Bad Name"):
        backend.graph.set_ontology(graph_ids=[GRAPH], entities={"Bad Name": unsafe}, edges=None)


def test_llm_and_embedder_respect_their_concurrency_caps(monkeypatch):
    from graphiti_core.embedder.openai import OpenAIEmbedderConfig
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

    peak = {"llm": 0, "embed": 0}
    live = {"llm": 0, "embed": 0}

    async def track(kind):
        live[kind] += 1
        peak[kind] = max(peak[kind], live[kind])
        await asyncio.sleep(0.005)
        live[kind] -= 1

    async def fake_generate(self, *args, **kwargs):
        await track("llm")

    formats = []

    async def fake_embed(**kwargs):
        formats.append(kwargs["encoding_format"])
        await track("embed")
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.0] * 2000)])

    monkeypatch.setattr(OpenAIGenericClient, "generate_response", fake_generate)

    async def run():
        llm = graphiti_backend.CappedLLMClient(3, config=LLMConfig(api_key="x", model="m"))
        embedder = graphiti_backend.CappedEmbedder(
            2,
            config=OpenAIEmbedderConfig(api_key="x", embedding_model="m", embedding_dim=768),
            client=SimpleNamespace(embeddings=SimpleNamespace(create=fake_embed)),
        )
        assert len(await embedder.create("x")) == 768
        await asyncio.gather(
            *(llm.generate_response([]) for _ in range(9)),
            *(embedder.create_batch([]) for _ in range(9)),
        )

    asyncio.run(run())
    assert peak == {"llm": 3, "embed": 2}
    assert set(formats) == {"float"}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('{"$defs": {}, "type": "object"}\n{"a": 2}', {"a": 2}),
        ('{"a": 1} trailing words', {"a": 1}),
    ],
)
def test_llm_replies_with_extra_content_keep_the_answer(raw, expected):
    import json

    assert json.loads(graphiti_backend.CappedLLMClient._strip_code_fences(raw)) == expected


def test_llm_retries_when_a_reply_does_not_match_the_model(monkeypatch):
    from graphiti_core.llm_client.config import LLMConfig
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.prompts.models import Message
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    replies = [{"$defs": {}, "type": "object"}, {"value": 7, "Person": {"role": None}}]
    prompts = []

    async def fake_generate(self, messages, *args, **kwargs):
        prompts.append(messages[-1].content)
        messages[-1].content += " mutated by the parent client"
        return replies.pop(0)

    monkeypatch.setattr(OpenAIGenericClient, "generate_response", fake_generate)

    async def run():
        llm = graphiti_backend.CappedLLMClient(1, config=LLMConfig(api_key="x", model="m"))
        return await llm.generate_response([Message(role="user", content="q")], Answer)

    assert asyncio.run(run()) == {"value": 7}
    assert prompts[0] == "q" and "mutated" not in prompts[1] and "not the schema" in prompts[1]


def test_get_zep_client_returns_graphiti_when_selected(monkeypatch, backend):
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    monkeypatch.setattr(graphiti_backend, "get_graphiti_backend", lambda: backend)
    assert zep.get_zep_client() is backend


def test_get_zep_client_still_returns_the_sdk_for_zep():
    zep.clear_zep_client_cache()
    assert isinstance(zep.get_zep_client("key"), Zep)


def test_neo4j_outage_is_retryable_for_reads():
    assert zep.is_retryable_zep_error(ServiceUnavailable("down"))


def test_ingestion_timeout_follows_the_backend(monkeypatch):
    assert zep.ingestion_wait_timeout(600) == 600
    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    monkeypatch.setattr(Config, "GRAPHITI_INGESTION_TIMEOUT_SECONDS", 7200)
    assert zep.ingestion_wait_timeout(600) == 7200


def test_validate_requires_each_backends_own_settings(monkeypatch):
    for name in ("ZEP_API_URL",):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Config, "LLM_API_KEY", "k")
    monkeypatch.setattr(Config, "DEBUG", False)

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "graphiti")
    for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "EMBEDDING_MODEL_NAME"):
        monkeypatch.setattr(Config, name, None)
    monkeypatch.setattr(Config, "ZEP_API_KEY", None)
    errors = Config.validate()
    assert any("NEO4J_URI" in e for e in errors) and not any("ZEP_API_KEY" in e for e in errors)

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "zep")
    errors = Config.validate()
    assert any("ZEP_API_KEY" in e for e in errors) and not any("NEO4J" in e for e in errors)

    monkeypatch.setattr(Config, "GRAPH_BACKEND", "other")
    assert any("GRAPH_BACKEND" in e for e in Config.validate())
