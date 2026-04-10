"""Tests for infermark.client — SSE parsing and request building."""

import json

from infermark.client import (
    _build_headers,
    _build_payload,
    _extract_token_from_chunk,
    _parse_sse_line,
)


class TestBuildPayload:
    def test_streaming(self):
        p = _build_payload("llama", "Hello", 100, True)
        assert p["model"] == "llama"
        assert p["messages"] == [{"role": "user", "content": "Hello"}]
        assert p["max_tokens"] == 100
        assert p["stream"] is True

    def test_non_streaming(self):
        p = _build_payload("gpt-4", "Hi", 50, False)
        assert p["stream"] is False

    def test_extra_body(self):
        p = _build_payload("m", "p", 10, True, {"temperature": 0.5})
        assert p["temperature"] == 0.5

    def test_extra_body_none(self):
        p = _build_payload("m", "p", 10, True, None)
        assert "temperature" not in p


class TestBuildHeaders:
    def test_with_key(self):
        h = _build_headers("sk-123")
        assert h["Authorization"] == "Bearer sk-123"
        assert h["Content-Type"] == "application/json"

    def test_without_key(self):
        h = _build_headers("")
        assert "Authorization" not in h


class TestParseSSELine:
    def test_valid_data(self):
        line = 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
        result = _parse_sse_line(line)
        assert result is not None
        assert result["choices"][0]["delta"]["content"] == "Hi"

    def test_done(self):
        assert _parse_sse_line("data: [DONE]") is None

    def test_empty_line(self):
        assert _parse_sse_line("") is None

    def test_comment(self):
        assert _parse_sse_line(": comment") is None

    def test_non_data_prefix(self):
        assert _parse_sse_line("event: message") is None

    def test_malformed_json(self):
        assert _parse_sse_line("data: {bad}") is None

    def test_whitespace(self):
        assert _parse_sse_line("   ") is None


class TestExtractToken:
    def test_content(self):
        chunk = {"choices": [{"delta": {"content": "Hello"}}]}
        assert _extract_token_from_chunk(chunk) == "Hello"

    def test_no_content(self):
        chunk = {"choices": [{"delta": {"role": "assistant"}}]}
        assert _extract_token_from_chunk(chunk) is None

    def test_empty_choices(self):
        chunk = {"choices": []}
        assert _extract_token_from_chunk(chunk) is None

    def test_no_choices(self):
        chunk = {}
        assert _extract_token_from_chunk(chunk) is None

    def test_empty_content(self):
        chunk = {"choices": [{"delta": {"content": ""}}]}
        assert _extract_token_from_chunk(chunk) == ""
