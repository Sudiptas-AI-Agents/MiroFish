"""Graphiti implementation of the GraphBackend surface defined by the Zep SDK.

Every MiroFish call site speaks the Zep client shape, so this module mirrors the
part of it they use instead of touching ~10 services. That keeps upstream merges
small. Unlike Zep Cloud, Neo4j Community has one database and no per-login
permissions: isolation is the ``mirofish_`` group_id prefix enforced here.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid as uuidlib
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Coroutine, Optional

# Graphiti reads this at import time; the homelab sends no telemetry.
os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.errors import GroupsEdgesNotFoundError, NodeNotFoundError
from graphiti_core.helpers import SAFE_CYPHER_IDENTIFIER_PATTERN
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.nodes import EntityNode, EpisodeType, Node
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_RRF,
    NODE_HYBRID_SEARCH_RRF,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, create_model
from zep_cloud import NotFoundError

from ..config import Config
from .llm_http import async_rate_limit_retry_client
from .logger import get_logger

logger = get_logger("mirofish.graphiti")

GRAPH_ID_PREFIX = "mirofish_"
_READ_TIMEOUT_SECONDS = 120.0
_LLM_VALIDATION_ATTEMPTS = 5
_NEXT_CURSOR_HEADER = "zep-next-cursor"


def _require_namespace(graph_id: str) -> str:
    """Refuse anything outside MiroFish's slice of the shared database."""

    if not isinstance(graph_id, str) or not graph_id.startswith(GRAPH_ID_PREFIX):
        raise ValueError(f"graph_id must start with {GRAPH_ID_PREFIX}: {graph_id!r}")
    return graph_id


def _not_found(what: str) -> NotFoundError:
    return NotFoundError(body={"message": f"{what} not found"})


def _uuid_of(item: Any) -> str:
    return getattr(item, "uuid_", None) or item.uuid


def _zep_node(node: Any) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=_uuid_of(node),
        name=node.name,
        labels=list(node.labels or []),
        summary=node.summary or "",
        attributes=dict(node.attributes or {}),
        created_at=node.created_at,
    )


def _zep_edge(edge: Any) -> SimpleNamespace:
    return SimpleNamespace(
        uuid_=_uuid_of(edge),
        name=edge.name,
        fact=edge.fact,
        source_node_uuid=edge.source_node_uuid,
        target_node_uuid=edge.target_node_uuid,
        attributes=dict(edge.attributes or {}),
        created_at=edge.created_at,
        valid_at=edge.valid_at,
        invalid_at=edge.invalid_at,
        expired_at=edge.expired_at,
        episodes=[str(episode) for episode in (edge.episodes or [])],
    )


def _describe_model(model: type[BaseModel]) -> dict[str, Any]:
    return {
        "doc": model.__doc__ or model.__name__,
        "fields": {
            name: info.description or name for name, info in model.model_fields.items()
        },
    }


def _graphiti_model(name: str, spec: dict[str, Any]) -> type[BaseModel]:
    reserved = EntityNode.model_fields.keys()
    fields: dict[str, Any] = {
        (f"entity_{field}" if field in reserved else field): (
            Optional[str],
            Field(default=None, description=description),
        )
        for field, description in spec["fields"].items()
    }
    return create_model(name, __doc__=spec["doc"], **fields)


def ontology_to_graphiti(ontology: dict[str, Any]) -> dict[str, Any]:
    """Turn a stored ontology into Graphiti's add_episode keyword arguments."""

    entity_types = {
        name: _graphiti_model(name, spec)
        for name, spec in ontology.get("entities", {}).items()
    }
    edge_types = {}
    edge_type_map: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name, spec in ontology.get("edges", {}).items():
        edge_types[name] = _graphiti_model(name, spec)
        for source, target in spec["source_targets"]:
            edge_type_map[(source, target)].append(name)
    return {
        "entity_types": entity_types or None,
        "edge_types": edge_types or None,
        "edge_type_map": dict(edge_type_map) or None,
    }


class CappedLLMClient(OpenAIGenericClient):
    """Keeps Graphiti inside MiroFish's share of the LiteLLM key's request limit."""

    def __init__(self, max_concurrency: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def generate_response(
        self, messages: list[Any], response_model: Any = None, *args: Any, **kwargs: Any
    ) -> dict[str, Any]:
        # Graphiti validates replies only after this returns, with no retry, and
        # these models sometimes return the schema itself instead of the data.
        nudge = ""
        for attempt in range(1, _LLM_VALIDATION_ATTEMPTS + 1):
            # The parent appends to the prompt in place, so each try gets a copy.
            attempt_messages = [message.model_copy() for message in messages]
            if nudge and attempt_messages:
                attempt_messages[-1].content += nudge
            async with self._semaphore:
                reply = await super().generate_response(
                    attempt_messages, response_model, *args, **kwargs
                )
            if response_model is None:
                return reply
            try:
                # Dropping unknown keys matters: Graphiti stores attribute replies
                # as Neo4j properties, which cannot hold a stray nested object.
                return response_model.model_validate(reply).model_dump()
            except ValidationError:
                if attempt == _LLM_VALIDATION_ATTEMPTS:
                    raise
                logger.warning("LLM reply did not match %s, retrying", response_model.__name__)
                nudge = "\n\nReturn the extracted data as JSON, not the schema."
        raise AssertionError("unreachable")

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Keep the last JSON value: some models echo the schema before answering."""

        stripped = OpenAIGenericClient._strip_code_fences(text)
        decoder = json.JSONDecoder()
        start, end, position = 0, len(stripped), 0
        try:
            while position < len(stripped):
                _, stop = decoder.raw_decode(stripped, position)
                start, end, position = position, stop, stop
                while position < len(stripped) and stripped[position].isspace():
                    position += 1
        except json.JSONDecodeError:
            pass  # trailing prose ends the scan; the last value found stands
        return stripped[start:end]


class CappedEmbedder(OpenAIEmbedder):
    """The embedding service is CPU-bound and shared with other apps."""

    def __init__(self, max_concurrency: int, **kwargs: Any):
        super().__init__(**kwargs)
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def _embed(self, input_data: Any) -> list[list[float]]:
        async with self._semaphore:
            # The OpenAI SDK defaults to base64, which this model server rejects.
            result = await self.client.embeddings.create(
                input=input_data,
                model=self.config.embedding_model,
                encoding_format="float",
            )
        return [item.embedding[: self.config.embedding_dim] for item in result.data]

    async def create(self, input_data: Any) -> list[float]:
        return (await self._embed(input_data))[0]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return await self._embed(input_data_list)


class _KeepOrderReranker(CrossEncoderClient):
    """Search uses RRF only; Graphiti still requires a client at construction."""

    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(passage, 1.0) for passage in passages]


class Neo4jGraphitiStore:
    """The only code that talks to Graphiti and Neo4j. All methods run on one loop."""

    def __init__(self, graphiti: Graphiti):
        self._graphiti = graphiti
        self._driver = graphiti.driver

    async def setup(self) -> None:
        await self._graphiti.build_indices_and_constraints()

    async def save_graph(self, graph_id: str, name: str, description: str) -> None:
        await self._driver.execute_query(
            "MERGE (g:MirofishGraph {graph_id: $graph_id}) "
            "SET g.name = $name, g.description = $description",
            params={"graph_id": graph_id, "name": name, "description": description},
        )

    async def load_graph(self, graph_id: str) -> dict[str, Any] | None:
        records, _, _ = await self._driver.execute_query(
            "MATCH (g:MirofishGraph {graph_id: $graph_id}) "
            "RETURN g.name AS name, g.description AS description, g.ontology AS ontology",
            params={"graph_id": graph_id},
        )
        if not records:
            return None
        record = records[0]
        return {
            "name": record["name"],
            "description": record["description"],
            "ontology": json.loads(record["ontology"]) if record["ontology"] else None,
        }

    async def save_ontology(self, graph_id: str, ontology: dict[str, Any]) -> None:
        await self._driver.execute_query(
            "MATCH (g:MirofishGraph {graph_id: $graph_id}) SET g.ontology = $ontology",
            params={"graph_id": graph_id, "ontology": json.dumps(ontology)},
        )

    async def delete_graph(self, graph_id: str) -> None:
        # Node.delete_by_group_id removes one group only; Graphiti's clear_data
        # must never be used here because it can wipe the shared database.
        await Node.delete_by_group_id(self._driver, graph_id)
        await self._driver.execute_query(
            "MATCH (g:MirofishGraph {graph_id: $graph_id}) DELETE g",
            params={"graph_id": graph_id},
        )

    async def add_episode(
        self,
        graph_id: str,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime,
        ontology: dict[str, Any] | None,
    ) -> None:
        await self._graphiti.add_episode(
            name=name,
            episode_body=body,
            source_description=source_description,
            reference_time=reference_time,
            source=EpisodeType.text,
            group_id=graph_id,
            **(ontology_to_graphiti(ontology) if ontology else {}),
        )

    async def search(
        self, graph_id: str, query: str, limit: int, scope: str
    ) -> tuple[list[Any], list[Any]]:
        recipe = EDGE_HYBRID_SEARCH_RRF if scope == "edges" else NODE_HYBRID_SEARCH_RRF
        results = await self._graphiti.search_(
            query,
            config=recipe.model_copy(update={"limit": limit}),
            group_ids=[graph_id],
        )
        return results.edges, results.nodes

    async def nodes_page(self, graph_id: str, limit: int, cursor: str | None) -> list[Any]:
        return await EntityNode.get_by_group_ids(
            self._driver, [graph_id], limit=limit, uuid_cursor=cursor
        )

    async def edges_page(self, graph_id: str, limit: int, cursor: str | None) -> list[Any]:
        try:
            return await EntityEdge.get_by_group_ids(
                self._driver, [graph_id], limit=limit, uuid_cursor=cursor
            )
        except GroupsEdgesNotFoundError:
            return []

    async def get_node(self, node_uuid: str) -> Any | None:
        try:
            return await EntityNode.get_by_uuid(self._driver, node_uuid)
        except NodeNotFoundError:
            return None

    async def node_edges(self, node_uuid: str) -> list[Any]:
        return await EntityEdge.get_by_node_uuid(self._driver, node_uuid)


class GraphitiBackend:
    """Synchronous, Zep-shaped facade over an async Graphiti store."""

    def __init__(self, store: Any):
        self._store = store
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        # ponytail: episode and batch progress is process-local, like every
        # other MiroFish task. A restart mid-ingest reads as a failed batch and
        # the owner rebuilds; persist it if builds ever outlive deploys.
        self._episodes: dict[str, SimpleNamespace] = {}
        self._batches: dict[str, SimpleNamespace] = {}
        self._group_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._ontologies: dict[str, dict[str, Any] | None] = {}

        raw_nodes = SimpleNamespace(get_by_graph_id=self._raw_nodes)
        raw_edges = SimpleNamespace(get_by_graph_id=self._raw_edges)
        self.graph = SimpleNamespace(
            create=self._create,
            get=self._get,
            delete=self._delete,
            set_ontology=self._set_ontology,
            add=self._add,
            search=self._search,
            node=SimpleNamespace(
                get=self._node_get,
                get_edges=self._node_get_edges,
                with_raw_response=raw_nodes,
            ),
            edge=SimpleNamespace(with_raw_response=raw_edges),
            episode=SimpleNamespace(get=self._episode_get),
        )
        self.batch = SimpleNamespace(
            create=self._batch_create,
            add=self._batch_add,
            process=self._batch_process,
            get=self._batch_get,
            list=self._batch_list,
            list_items=self._batch_list_items,
        )

    def _run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(
            _READ_TIMEOUT_SECONDS
        )

    def _submit(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    # --- graphs ---

    def _create(self, *, graph_id: str, name: str = "", description: str = "") -> None:
        _require_namespace(graph_id)
        self._run(self._store.save_graph(graph_id, name, description))

    def _load(self, graph_id: str) -> dict[str, Any]:
        graph = self._run(self._store.load_graph(_require_namespace(graph_id)))
        if graph is None:
            raise _not_found(f"Graph {graph_id}")
        return graph

    def _get(self, graph_id: str) -> SimpleNamespace:
        graph = self._load(graph_id)
        return SimpleNamespace(
            graph_id=graph_id, name=graph["name"], description=graph["description"]
        )

    def _delete(self, graph_id: str) -> None:
        self._load(graph_id)
        self._run(self._store.delete_graph(graph_id))
        self._ontologies.pop(graph_id, None)

    def _set_ontology(
        self,
        *,
        graph_ids: list[str],
        entities: dict[str, type[BaseModel]] | None = None,
        edges: dict[str, tuple[type[BaseModel], list[Any]]] | None = None,
    ) -> None:
        for name in list(entities or {}) + list(edges or {}):
            if not SAFE_CYPHER_IDENTIFIER_PATTERN.match(name):
                raise ValueError(f"Ontology type name is not a safe Neo4j label: {name}")
        ontology = {
            "entities": {
                name: _describe_model(model) for name, model in (entities or {}).items()
            },
            "edges": {
                name: {
                    **_describe_model(model),
                    "source_targets": [[st.source, st.target] for st in source_targets],
                }
                for name, (model, source_targets) in (edges or {}).items()
            },
        }
        for graph_id in graph_ids:
            self._load(graph_id)
            self._run(self._store.save_ontology(graph_id, ontology))
            self._ontologies[graph_id] = ontology

    def _ontology(self, graph_id: str) -> dict[str, Any] | None:
        if graph_id not in self._ontologies:
            self._ontologies[graph_id] = self._load(graph_id)["ontology"]
        return self._ontologies[graph_id]

    # --- ingestion ---

    async def _ingest(
        self,
        episode_uuid: str,
        graph_id: str,
        data: str,
        source_description: str,
        reference_time: datetime,
        ontology: dict[str, Any] | None,
    ) -> None:
        state = self._episodes[episode_uuid]
        # Graphiti requires episodes of one group to be added sequentially.
        async with self._group_locks[graph_id]:
            try:
                await self._store.add_episode(
                    graph_id,
                    f"episode-{episode_uuid}",
                    data,
                    source_description or "MiroFish",
                    reference_time,
                    ontology,
                )
                state.processed = True
            except Exception as error:
                logger.error("Graphiti ingestion failed for %s: %s", graph_id, error)
                state.error = str(error) or type(error).__name__

    def _new_episode(self) -> str:
        episode_uuid = str(uuidlib.uuid4())
        self._episodes[episode_uuid] = SimpleNamespace(processed=False, error=None)
        return episode_uuid

    def _add(
        self,
        *,
        graph_id: str,
        type: str = "text",
        data: str,
        created_at: str | None = None,
        source_description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        _require_namespace(graph_id)
        reference_time = (
            datetime.fromisoformat(created_at) if created_at else datetime.now(timezone.utc)
        )
        ontology = self._ontology(graph_id)
        episode_uuid = self._new_episode()
        self._submit(
            self._ingest(
                episode_uuid, graph_id, data, source_description or "", reference_time, ontology
            )
        )
        return SimpleNamespace(uuid_=episode_uuid, processed=False)

    def _episode_get(self, uuid_: str) -> SimpleNamespace:
        state = self._episodes.get(uuid_)
        if state is None:
            raise _not_found(f"Episode {uuid_}")
        if state.error:
            raise RuntimeError(f"Graphiti ingestion failed: {state.error}")
        return SimpleNamespace(uuid_=uuid_, processed=state.processed)

    # --- batches (Zep's Batch API is emulated on top of episodes) ---

    def _batch_create(self, *, metadata: dict[str, Any] | None = None) -> SimpleNamespace:
        batch = SimpleNamespace(
            batch_id=uuidlib.uuid4().hex,
            metadata=metadata or {},
            status="draft",
            items=[],
            progress=SimpleNamespace(percent_complete=0.0, succeeded_items=0),
        )
        self._batches[batch.batch_id] = batch
        return batch

    def _batch_add(self, *, batch_id: str, items: list[Any]) -> list[SimpleNamespace]:
        batch = self._batches[batch_id]
        added = []
        for item in items:
            record = SimpleNamespace(
                sequence_index=len(batch.items),
                status="pending",
                episode_uuid=self._new_episode(),
                source_uuid=None,
                error=None,
                graph_id=_require_namespace(item.graph_id),
                data=item.data,
                source_description=item.source_description,
            )
            batch.items.append(record)
            added.append(record)
        return added

    def _batch_process(self, *, batch_id: str) -> None:
        batch = self._batches[batch_id]
        ontologies = {
            graph_id: self._ontology(graph_id)
            for graph_id in {item.graph_id for item in batch.items}
        }
        batch.status = "queued"
        self._submit(self._run_batch(batch, ontologies))

    async def _run_batch(self, batch: SimpleNamespace, ontologies: dict[str, Any]) -> None:
        batch.status = "processing"
        total = len(batch.items)
        for done, item in enumerate(batch.items, start=1):
            await self._ingest(
                item.episode_uuid,
                item.graph_id,
                item.data,
                item.source_description or "",
                datetime.now(timezone.utc),
                ontologies[item.graph_id],
            )
            item.data = None  # chunk text is not needed once ingested
            state = self._episodes[item.episode_uuid]
            if state.error:
                item.status, item.error = "failed", state.error
            else:
                item.status = "succeeded"
                batch.progress.succeeded_items += 1
            batch.progress.percent_complete = 100.0 * done / total
        succeeded = batch.progress.succeeded_items
        batch.status = "succeeded" if succeeded == total else ("partial" if succeeded else "failed")

    def _batch_get(self, *, batch_id: str) -> SimpleNamespace:
        batch = self._batches.get(batch_id)
        if batch is not None:
            return batch
        return SimpleNamespace(
            batch_id=batch_id,
            status="failed",
            metadata={},
            items=[],
            progress=SimpleNamespace(percent_complete=0.0, succeeded_items=0),
        )

    def _batch_list(self, *, limit: int = 100, cursor: int | None = None) -> SimpleNamespace:
        return SimpleNamespace(batches=list(self._batches.values()), next_cursor=None)

    def _batch_list_items(
        self, *, batch_id: str, limit: int = 100, cursor: int | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(items=list(self._batch_get(batch_id=batch_id).items), next_cursor=None)

    # --- reads ---

    def _search(
        self,
        *,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
        reranker: str | None = None,
    ) -> SimpleNamespace:
        # ponytail: every reranker maps to RRF. Graphiti's cross-encoder needs
        # logprobs from a real OpenAI endpoint, which LiteLLM's models lack.
        if scope not in {"edges", "nodes"}:
            raise ValueError(f"Unsupported search scope: {scope}")
        edges, nodes = self._run(
            self._store.search(_require_namespace(graph_id), query, limit, scope)
        )
        return SimpleNamespace(
            edges=[_zep_edge(edge) for edge in edges],
            nodes=[_zep_node(node) for node in nodes],
        )

    def _page(self, fetch: Any, mapper: Any, graph_id: str, limit: int, cursor: str | None) -> Any:
        items = self._run(fetch(_require_namespace(graph_id), limit, cursor))
        headers = {_NEXT_CURSOR_HEADER: items[-1].uuid} if len(items) == limit else {}
        return SimpleNamespace(data=[mapper(item) for item in items], headers=headers)

    def _raw_nodes(self, graph_id: str, *, limit: int = 100, cursor: str | None = None) -> Any:
        return self._page(self._store.nodes_page, _zep_node, graph_id, limit, cursor)

    def _raw_edges(self, graph_id: str, *, limit: int = 100, cursor: str | None = None) -> Any:
        return self._page(self._store.edges_page, _zep_edge, graph_id, limit, cursor)

    def _node_get(self, uuid_: str) -> SimpleNamespace:
        node = self._run(self._store.get_node(uuid_))
        if node is None:
            raise _not_found(f"Node {uuid_}")
        return _zep_node(node)

    def _node_get_edges(self, node_uuid: str) -> list[SimpleNamespace]:
        return [_zep_edge(edge) for edge in self._run(self._store.node_edges(node_uuid))]


def _retrying_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=Config.LLM_API_KEY or "unset",
        base_url=Config.LLM_BASE_URL,
        http_client=async_rate_limit_retry_client(),
    )


@lru_cache(maxsize=1)
def get_graphiti_backend() -> GraphitiBackend:
    """Return the process-wide backend, creating Neo4j indexes on first use."""

    # Built off the backend's loop on purpose: on a running loop Neo4jDriver
    # schedules its own index build, which races the awaited one below.
    graphiti = Graphiti(
        graph_driver=Neo4jDriver(
            Config.NEO4J_URI,
            Config.NEO4J_USER,
            Config.NEO4J_PASSWORD,
            database=Config.NEO4J_DATABASE,
        ),
        llm_client=CappedLLMClient(
            Config.GRAPHITI_MAX_CONCURRENCY,
            config=LLMConfig(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                model=Config.LLM_MODEL_NAME,
                small_model=Config.LLM_MODEL_NAME,
            ),
            client=_retrying_openai_client(),
            # Measured 2026-11-08: these models answer json_schema requests in
            # prose through LiteLLM, but follow a schema given in the prompt.
            structured_output_mode="json_object",
        ),
        embedder=CappedEmbedder(
            Config.GRAPHITI_EMBED_CONCURRENCY,
            config=OpenAIEmbedderConfig(
                api_key=Config.LLM_API_KEY,
                base_url=Config.LLM_BASE_URL,
                embedding_model=Config.EMBEDDING_MODEL_NAME,
                embedding_dim=Config.EMBEDDING_DIM,
            ),
            client=_retrying_openai_client(),
        ),
        cross_encoder=_KeepOrderReranker(),
        max_coroutines=Config.GRAPHITI_MAX_CONCURRENCY,
    )
    store = Neo4jGraphitiStore(graphiti)
    backend = GraphitiBackend(store)
    backend._run(store.setup())
    return backend
