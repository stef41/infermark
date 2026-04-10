"""Backend abstraction for different LLM serving frameworks."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from infermark._types import RequestResult


class Backend(ABC):
    """Base class for LLM inference backends."""

    @abstractmethod
    def send_request(self, prompt: str, config: Dict[str, Any]) -> RequestResult:
        """Send a single inference request.

        Parameters
        ----------
        prompt:
            The user prompt to send.
        config:
            Backend-specific configuration (``max_tokens``, ``temperature``, etc.).

        Returns
        -------
        RequestResult
        """
        ...

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend."""
        ...


def _http_post(url: str, payload: Dict[str, Any], headers: Dict[str, str],
               timeout: float = 120.0) -> Dict[str, Any]:
    """Perform a synchronous HTTP POST and return the parsed JSON response."""
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


def _http_get(url: str, timeout: float = 10.0) -> Dict[str, Any]:
    """Perform a synchronous HTTP GET and return the parsed JSON response."""
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


class OpenAIBackend(Backend):
    """Backend for OpenAI-compatible APIs (default)."""

    def __init__(self, url: str, model: str, api_key: str = "") -> None:
        self.url = url.rstrip("/")
        self.model = model
        self.api_key = api_key

    @property
    def backend_name(self) -> str:
        return "openai"

    def send_request(self, prompt: str, config: Dict[str, Any]) -> RequestResult:
        endpoint = f"{self.url}/v1/chat/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.get("max_tokens", 256),
            "temperature": config.get("temperature", 1.0),
            "stream": False,
        }

        timeout = config.get("timeout", 120.0)
        start = time.perf_counter()
        try:
            resp = _http_post(endpoint, payload, headers, timeout=timeout)
            latency = time.perf_counter() - start

            output_tokens = 0
            usage = resp.get("usage", {})
            if usage:
                output_tokens = usage.get("completion_tokens", 0)

            return RequestResult(
                success=True,
                latency=latency,
                output_tokens=output_tokens,
                input_tokens=usage.get("prompt_tokens", 0),
            )
        except Exception as exc:
            return RequestResult(
                success=False,
                latency=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )


class VLLMBackend(Backend):
    """Backend for vLLM's native API."""

    def __init__(self, url: str, model: str) -> None:
        self.url = url.rstrip("/")
        self.model = model

    @property
    def backend_name(self) -> str:
        return "vllm"

    def send_request(self, prompt: str, config: Dict[str, Any]) -> RequestResult:
        endpoint = f"{self.url}/v1/completions"
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": config.get("max_tokens", 256),
            "temperature": config.get("temperature", 1.0),
            "stream": False,
        }
        # vLLM-specific parameters
        if "best_of" in config:
            payload["best_of"] = config["best_of"]
        if "presence_penalty" in config:
            payload["presence_penalty"] = config["presence_penalty"]
        if "frequency_penalty" in config:
            payload["frequency_penalty"] = config["frequency_penalty"]
        if "top_k" in config:
            payload["top_k"] = config["top_k"]
        if "use_beam_search" in config:
            payload["use_beam_search"] = config["use_beam_search"]

        timeout = config.get("timeout", 120.0)
        start = time.perf_counter()
        try:
            resp = _http_post(endpoint, payload, headers, timeout=timeout)
            latency = time.perf_counter() - start

            usage = resp.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)

            return RequestResult(
                success=True,
                latency=latency,
                output_tokens=output_tokens,
                input_tokens=usage.get("prompt_tokens", 0),
            )
        except Exception as exc:
            return RequestResult(
                success=False,
                latency=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )


class TGIBackend(Backend):
    """Backend for HuggingFace Text Generation Inference (TGI)."""

    def __init__(self, url: str, model: str = "") -> None:
        self.url = url.rstrip("/")
        self.model = model

    @property
    def backend_name(self) -> str:
        return "tgi"

    def send_request(self, prompt: str, config: Dict[str, Any]) -> RequestResult:
        endpoint = f"{self.url}/generate"
        headers: Dict[str, str] = {"Content-Type": "application/json"}

        parameters: Dict[str, Any] = {
            "max_new_tokens": config.get("max_tokens", 256),
            "temperature": config.get("temperature", 1.0),
        }
        if "top_k" in config:
            parameters["top_k"] = config["top_k"]
        if "top_p" in config:
            parameters["top_p"] = config["top_p"]
        if "repetition_penalty" in config:
            parameters["repetition_penalty"] = config["repetition_penalty"]

        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": parameters,
        }

        timeout = config.get("timeout", 120.0)
        start = time.perf_counter()
        try:
            resp = _http_post(endpoint, payload, headers, timeout=timeout)
            latency = time.perf_counter() - start

            # TGI returns generated_text and optionally details with token counts
            details = resp.get("details", {})
            output_tokens = 0
            if details:
                generated_tokens = details.get("generated_tokens", 0)
                output_tokens = generated_tokens

            return RequestResult(
                success=True,
                latency=latency,
                output_tokens=output_tokens,
            )
        except Exception as exc:
            return RequestResult(
                success=False,
                latency=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )


def detect_backend(url: str) -> str:
    """Try to detect which backend a URL uses.

    Probes ``/info`` (TGI), ``/health`` (TGI), and ``/v1/models`` (OpenAI/vLLM)
    to determine the backend type.

    Returns
    -------
    str
        One of ``"tgi"``, ``"vllm"``, ``"openai"``, or ``"unknown"``.
    """
    url = url.rstrip("/")

    # Check /info first (TGI specific)
    try:
        info = _http_get(f"{url}/info", timeout=5.0)
        # TGI /info returns model_id, and sometimes docker_label
        if "model_id" in info:
            return "tgi"
    except Exception:
        pass

    # Check /v1/models (OpenAI-compatible — both vLLM and OpenAI expose this)
    try:
        models = _http_get(f"{url}/v1/models", timeout=5.0)
        if "data" in models:
            # vLLM typically includes "object": "list" in models response
            # Check /health which is vLLM-specific
            try:
                _http_get(f"{url}/health", timeout=5.0)
                return "vllm"
            except Exception:
                return "openai"
    except Exception:
        pass

    # Check /health alone (could be vLLM without /v1/models accessible)
    try:
        _http_get(f"{url}/health", timeout=5.0)
        return "vllm"
    except Exception:
        pass

    return "unknown"
