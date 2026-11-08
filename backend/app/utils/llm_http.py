"""Wait out LiteLLM key-limit 429s so a burst never drops an LLM call.

Lives at the HTTP layer because three SDK stacks (MiroFish's OpenAI clients,
Graphiti and OASIS/camel) share one LiteLLM key, and the OpenAI SDK's own
max_retries would also multiply timeouts and 5xx failures.
"""

from __future__ import annotations

import asyncio
import os
import random
import re
import time
from typing import Any

import httpx
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

from .logger import get_logger

logger = get_logger("mirofish.llm_http")

_KEY_LIMIT_429 = re.compile(r"parallel|rate.?limit", re.IGNORECASE)
MAX_RETRY_DELAY_SECONDS = 30.0
RATE_LIMIT_MAX_WAIT_SECONDS = float(os.environ.get("LLM_RATE_LIMIT_MAX_WAIT_SECONDS", "300"))


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Full jitter, so callers that collided do not all return together."""

    delay = random.uniform(0.5, min(MAX_RETRY_DELAY_SECONDS, 2.0 ** (attempt + 1)))
    try:
        floor = float(retry_after) if retry_after else 0.0
    except ValueError:
        floor = 0.0
    return min(max(delay, floor + random.uniform(0, 1)), MAX_RETRY_DELAY_SECONDS)


def _should_wait(response: httpx.Response, started: float) -> bool:
    if not _KEY_LIMIT_429.search(response.text):
        return False  # e.g. an exhausted budget: waiting cannot fix it
    return time.monotonic() - started < RATE_LIMIT_MAX_WAIT_SECONDS


class RateLimitRetryTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport | None = None):
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        attempt = 0
        while True:
            response = self._inner.handle_request(request)
            if response.status_code != 429:
                return response
            response.read()
            if not _should_wait(response, started):
                return response
            delay = _retry_delay(attempt, response.headers.get("retry-after"))
            logger.warning("LiteLLM key limit hit, retrying in %.1fs", delay)
            response.close()
            time.sleep(delay)
            attempt += 1

    def close(self) -> None:
        self._inner.close()


class AsyncRateLimitRetryTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport | None = None):
        self._inner = inner or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        attempt = 0
        while True:
            response = await self._inner.handle_async_request(request)
            if response.status_code != 429:
                return response
            await response.aread()
            if not _should_wait(response, started):
                return response
            delay = _retry_delay(attempt, response.headers.get("retry-after"))
            logger.warning("LiteLLM key limit hit, retrying in %.1fs", delay)
            await response.aclose()
            await asyncio.sleep(delay)
            attempt += 1

    async def aclose(self) -> None:
        await self._inner.aclose()


def rate_limit_retry_client(inner: httpx.BaseTransport | None = None) -> httpx.Client:
    return DefaultHttpxClient(transport=RateLimitRetryTransport(inner))


def async_rate_limit_retry_client(
    inner: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    return DefaultAsyncHttpxClient(transport=AsyncRateLimitRetryTransport(inner))


def with_rate_limit_retry(model: Any, inner: Any = None) -> Any:
    """Give a camel model retrying clients; its kwargs cannot, as sync and async share them."""

    model._client = model._client.with_options(http_client=rate_limit_retry_client(inner))
    model._async_client = model._async_client.with_options(
        http_client=async_rate_limit_retry_client(inner)
    )
    return model
