"""Async HTTP client for benchmarking OpenAI-compatible endpoints."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from infermark._types import BenchmarkMode, RequestResult


def _build_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    stream: bool,
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the chat completions request payload."""
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if extra_body:
        payload.update(extra_body)
    return payload


def _build_headers(api_key: str) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single SSE data line, returning the JSON payload or None."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _extract_token_from_chunk(chunk: Dict[str, Any]) -> Optional[str]:
    """Extract the content delta from a streaming chunk."""
    choices = chunk.get("choices", [])
    if not choices:
        return None
    delta = choices[0].get("delta", {})
    return delta.get("content")


async def send_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> RequestResult:
    """Send a streaming request and measure TTFT and ITL."""
    start = time.perf_counter()
    first_token_time: Optional[float] = None
    last_token_time: Optional[float] = None
    itl: list[float] = []
    output_tokens = 0

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                body = b""
                async for raw_chunk in response.aiter_bytes():
                    body += raw_chunk
                return RequestResult(
                    success=False,
                    latency=time.perf_counter() - start,
                    error=f"HTTP {response.status_code}: {body[:500].decode(errors='replace')}",
                )

            async for line in response.aiter_lines():
                chunk = _parse_sse_line(line)
                if chunk is None:
                    continue
                token = _extract_token_from_chunk(chunk)
                if token is not None and token != "":
                    now = time.perf_counter()
                    output_tokens += 1
                    if first_token_time is None:
                        first_token_time = now
                    else:
                        if last_token_time is not None:
                            itl.append(now - last_token_time)
                    last_token_time = now

        end = time.perf_counter()
        ttft = (first_token_time - start) if first_token_time else None

        return RequestResult(
            success=True,
            latency=end - start,
            ttft=ttft,
            itl=itl,
            output_tokens=output_tokens,
        )

    except httpx.TimeoutException:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error="timeout",
        )
    except httpx.ConnectError as e:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error=f"connection_error: {e}",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error=f"unknown: {type(e).__name__}: {e}",
        )


async def send_non_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout: float,
) -> RequestResult:
    """Send a non-streaming request and measure latency."""
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload, headers=headers, timeout=timeout)
        end = time.perf_counter()

        if response.status_code != 200:
            return RequestResult(
                success=False,
                latency=end - start,
                error=f"HTTP {response.status_code}: {response.text[:500]}",
            )

        data = response.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return RequestResult(
            success=True,
            latency=end - start,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    except httpx.TimeoutException:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error="timeout",
        )
    except httpx.ConnectError as e:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error=f"connection_error: {e}",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            latency=time.perf_counter() - start,
            error=f"unknown: {type(e).__name__}: {e}",
        )


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    mode: BenchmarkMode,
    api_key: str = "",
    timeout: float = 120.0,
    extra_body: Optional[Dict[str, Any]] = None,
) -> RequestResult:
    """Send a single benchmark request."""
    stream = mode == BenchmarkMode.STREAMING
    payload = _build_payload(model, prompt, max_tokens, stream, extra_body)
    headers = _build_headers(api_key)
    endpoint = f"{url}/chat/completions"

    if stream:
        return await send_streaming_request(client, endpoint, payload, headers, timeout)
    else:
        return await send_non_streaming_request(client, endpoint, payload, headers, timeout)
