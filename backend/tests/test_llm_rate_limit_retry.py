"""LiteLLM key-limit 429s must wait and retry instead of dropping the call."""

import asyncio

import httpx
import pytest
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from openai import AsyncOpenAI, OpenAI, RateLimitError

from app.utils import llm_http

COMPLETION = {
    "id": "c", "object": "chat.completion", "created": 1, "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "ok"}}],
}
KEY_LIMIT = "Max parallel request limit reached for key"


def scripted(*statuses, message=KEY_LIMIT):
    """A fake LiteLLM answering with the given status codes, then success."""

    remaining = list(statuses)
    calls = []

    def handler(request):
        calls.append(request.content)
        if remaining:
            return httpx.Response(remaining.pop(0), json={"error": {"message": message}})
        return httpx.Response(200, json=COMPLETION)

    return httpx.MockTransport(handler), calls


@pytest.fixture
def sleeps(monkeypatch):
    waited = []

    async def fake_async_sleep(seconds):
        waited.append(seconds)

    monkeypatch.setattr(llm_http.time, "sleep", waited.append)
    monkeypatch.setattr(llm_http.asyncio, "sleep", fake_async_sleep)
    return waited


def sync_client(inner):
    return OpenAI(
        api_key="k", base_url="http://litellm/v1", max_retries=0,
        http_client=llm_http.rate_limit_retry_client(inner=inner),
    )


def ask(client):
    return client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])


@pytest.mark.parametrize("message", [KEY_LIMIT, "litellm.RateLimitError: rate limit exceeded"])
def test_sync_call_waits_through_key_limit_429s(sleeps, message):
    inner, calls = scripted(429, 429, 429, message=message)

    assert ask(sync_client(inner)).choices[0].message.content == "ok"

    assert len(calls) == 4 and len(sleeps) == 3
    assert len(set(calls)) == 1  # the same request is replayed


def test_async_call_waits_through_key_limit_429s(sleeps):
    inner, calls = scripted(429, 429)
    client = AsyncOpenAI(
        api_key="k", base_url="http://litellm/v1", max_retries=0,
        http_client=llm_http.async_rate_limit_retry_client(inner=inner),
    )

    reply = asyncio.run(
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
    )

    assert reply.choices[0].message.content == "ok" and len(calls) == 3 and len(sleeps) == 2


def test_delays_are_jittered(monkeypatch):
    delays = {llm_http._retry_delay(4, None) for _ in range(20)}
    assert len(delays) > 1
    assert all(0 < delay <= llm_http.MAX_RETRY_DELAY_SECONDS for delay in delays)


def test_retry_after_is_a_floor_for_the_delay():
    assert llm_http._retry_delay(0, "7") >= 7


def test_a_different_429_is_not_retried(sleeps):
    inner, calls = scripted(429, message="Budget has been exceeded for this key")

    with pytest.raises(RateLimitError):
        ask(sync_client(inner))

    assert len(calls) == 1 and sleeps == []


def test_other_errors_are_not_retried_here(sleeps):
    inner, calls = scripted(500)

    with pytest.raises(Exception):
        ask(sync_client(inner))

    assert len(calls) == 1 and sleeps == []


def test_gives_up_only_after_the_wait_budget(monkeypatch, sleeps):
    monkeypatch.setattr(llm_http, "RATE_LIMIT_MAX_WAIT_SECONDS", 0)
    inner, calls = scripted(429, 429)

    with pytest.raises(RateLimitError):
        ask(sync_client(inner))

    assert len(calls) == 1


def test_simulation_model_retries_on_its_async_client(sleeps):
    inner, calls = scripted(429, 429)
    model = ModelFactory.create(
        model_platform=ModelPlatformType.OPENAI, model_type="m",
        api_key="k", url="http://litellm/v1",
    )

    llm_http.with_rate_limit_retry(model, inner=inner)

    reply = asyncio.run(
        model._async_client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "q"}]
        )
    )
    assert reply.choices[0].message.content == "ok" and len(calls) == 3


def test_graphiti_clients_use_the_retrying_http_client():
    from app.utils import graphiti_backend

    assert isinstance(
        graphiti_backend._retrying_openai_client()._client._transport,
        llm_http.AsyncRateLimitRetryTransport,
    )
