"""Tests for infermark.backends module."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch
from http.client import HTTPResponse
from io import BytesIO

import pytest

from infermark.backends import (
    Backend,
    OpenAIBackend,
    VLLMBackend,
    TGIBackend,
    detect_backend,
    _http_post,
    _http_get,
)
from infermark._types import RequestResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_urlopen(response_body: Dict[str, Any], status: int = 200):
    """Return a context-manager mock for urllib.request.urlopen."""
    body = json.dumps(response_body).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.status = status
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ---------------------------------------------------------------------------
# OpenAIBackend
# ---------------------------------------------------------------------------

class TestOpenAIBackend:
    def test_backend_name(self) -> None:
        b = OpenAIBackend("http://localhost:8000", "gpt-4")
        assert b.backend_name == "openai"

    def test_url_trailing_slash_stripped(self) -> None:
        b = OpenAIBackend("http://localhost:8000/", "gpt-4")
        assert b.url == "http://localhost:8000"

    @patch("infermark.backends.urlopen")
    def test_send_request_success(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _mock_urlopen({
            "choices": [{"message": {"content": "hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        })
        b = OpenAIBackend("http://localhost:8000", "gpt-4", api_key="sk-test")
        result = b.send_request("Say hi", {"max_tokens": 64})
        assert result.success is True
        assert result.output_tokens == 10
        assert result.input_tokens == 5
        assert result.latency > 0

    @patch("infermark.backends.urlopen", side_effect=Exception("connection refused"))
    def test_send_request_failure(self, mock_open: MagicMock) -> None:
        b = OpenAIBackend("http://localhost:8000", "gpt-4")
        result = b.send_request("hi", {})
        assert result.success is False
        assert "connection refused" in (result.error or "")

    def test_api_key_stored(self) -> None:
        b = OpenAIBackend("http://x", "m", api_key="my-key")
        assert b.api_key == "my-key"


# ---------------------------------------------------------------------------
# VLLMBackend
# ---------------------------------------------------------------------------

class TestVLLMBackend:
    def test_backend_name(self) -> None:
        b = VLLMBackend("http://localhost:8000", "llama-7b")
        assert b.backend_name == "vllm"

    @patch("infermark.backends.urlopen")
    def test_send_request_success(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _mock_urlopen({
            "choices": [{"text": "Generated text here"}],
            "usage": {"prompt_tokens": 8, "completion_tokens": 15},
        })
        b = VLLMBackend("http://localhost:8000", "llama-7b")
        result = b.send_request("Say hi", {"max_tokens": 64, "best_of": 2})
        assert result.success is True
        assert result.output_tokens == 15

    @patch("infermark.backends.urlopen")
    def test_vllm_specific_params(self, mock_open: MagicMock) -> None:
        """Verify vLLM-specific params are included in the payload."""
        mock_open.return_value = _mock_urlopen({
            "choices": [{"text": "ok"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })
        b = VLLMBackend("http://localhost:8000", "llama-7b")
        b.send_request("test", {
            "best_of": 3,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.2,
            "top_k": 50,
            "use_beam_search": True,
        })
        # Verify the POST payload
        call_args = mock_open.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["best_of"] == 3
        assert payload["presence_penalty"] == 0.5
        assert payload["frequency_penalty"] == 0.2
        assert payload["top_k"] == 50
        assert payload["use_beam_search"] is True

    @patch("infermark.backends.urlopen", side_effect=Exception("timeout"))
    def test_send_request_failure(self, mock_open: MagicMock) -> None:
        b = VLLMBackend("http://localhost:8000", "llama-7b")
        result = b.send_request("hi", {})
        assert result.success is False


# ---------------------------------------------------------------------------
# TGIBackend
# ---------------------------------------------------------------------------

class TestTGIBackend:
    def test_backend_name(self) -> None:
        b = TGIBackend("http://localhost:8080")
        assert b.backend_name == "tgi"

    @patch("infermark.backends.urlopen")
    def test_send_request_success(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _mock_urlopen({
            "generated_text": "Hello world!",
            "details": {"generated_tokens": 12},
        })
        b = TGIBackend("http://localhost:8080", "bigscience/bloom")
        result = b.send_request("Say hello", {"max_tokens": 64})
        assert result.success is True
        assert result.output_tokens == 12

    @patch("infermark.backends.urlopen")
    def test_tgi_parameters_forwarded(self, mock_open: MagicMock) -> None:
        mock_open.return_value = _mock_urlopen({
            "generated_text": "ok",
            "details": {"generated_tokens": 1},
        })
        b = TGIBackend("http://localhost:8080")
        b.send_request("test", {
            "max_tokens": 128,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "repetition_penalty": 1.2,
        })
        call_args = mock_open.call_args
        req = call_args[0][0]
        payload = json.loads(req.data.decode())
        assert payload["inputs"] == "test"
        params = payload["parameters"]
        assert params["max_new_tokens"] == 128
        assert params["temperature"] == 0.7
        assert params["top_k"] == 40
        assert params["top_p"] == 0.9
        assert params["repetition_penalty"] == 1.2

    @patch("infermark.backends.urlopen", side_effect=Exception("bad"))
    def test_send_request_failure(self, mock_open: MagicMock) -> None:
        b = TGIBackend("http://localhost:8080")
        result = b.send_request("hi", {})
        assert result.success is False


# ---------------------------------------------------------------------------
# detect_backend
# ---------------------------------------------------------------------------

class TestDetectBackend:
    @patch("infermark.backends._http_get")
    def test_detect_tgi(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"model_id": "bigscience/bloom"}
        assert detect_backend("http://localhost:8080") == "tgi"

    @patch("infermark.backends._http_get")
    def test_detect_vllm(self, mock_get: MagicMock) -> None:
        def side_effect(url: str, timeout: float = 10.0) -> Dict[str, Any]:
            if "/info" in url:
                raise Exception("not found")
            if "/v1/models" in url:
                return {"data": [{"id": "llama"}], "object": "list"}
            if "/health" in url:
                return {}
            raise Exception("not found")
        mock_get.side_effect = side_effect
        assert detect_backend("http://localhost:8000") == "vllm"

    @patch("infermark.backends._http_get")
    def test_detect_openai(self, mock_get: MagicMock) -> None:
        def side_effect(url: str, timeout: float = 10.0) -> Dict[str, Any]:
            if "/info" in url:
                raise Exception("not found")
            if "/v1/models" in url:
                return {"data": [{"id": "gpt-4"}], "object": "list"}
            if "/health" in url:
                raise Exception("not found")
            raise Exception("not found")
        mock_get.side_effect = side_effect
        assert detect_backend("http://api.openai.com") == "openai"

    @patch("infermark.backends._http_get", side_effect=Exception("unreachable"))
    def test_detect_unknown(self, mock_get: MagicMock) -> None:
        assert detect_backend("http://localhost:9999") == "unknown"
