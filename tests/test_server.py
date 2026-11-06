"""Server plumbing: greedy payloads, tolerant response parsing, clear errors."""

from __future__ import annotations

import pytest

from src.server import (
    LlamaServer,
    ServerError,
    build_payload,
    free_port,
    parse_response,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def test_payload_is_greedy_and_stateless():
    payload = build_payload(MESSAGES, max_tokens=64)
    assert payload == {
        "messages": MESSAGES,
        "temperature": 0.0,
        "max_tokens": 64,
        "stream": False,
    }


def test_payload_can_request_json_mode():
    payload = build_payload(MESSAGES, response_format={"type": "json_object"})
    assert payload["response_format"] == {"type": "json_object"}


def test_parse_response_keeps_usage_and_timings():
    parsed = parse_response({
        "choices": [{"message": {"content": " hi "}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 3},
        "timings": {"predicted_per_second": 12.0},
    })
    assert parsed["text"] == "hi"
    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"]["completion_tokens"] == 3
    assert parsed["timings"]["predicted_per_second"] == 12.0


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]},
                                     {"choices": [{"message": {"content": " "}}]}])
def test_parse_response_rejects_empty(payload):
    with pytest.raises(ServerError):
        parse_response(payload)


def test_command_has_the_t4_defaults(tmp_path):
    server = LlamaServer(model_path=tmp_path / "m.gguf")
    command = server.command(8080)
    assert "-ngl" in command and command[command.index("-ngl") + 1] == "99"
    assert "--jinja" in command
    assert "127.0.0.1" in command


def test_missing_binary_is_actionable(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"")
    server = LlamaServer(model_path=model, server_bin=tmp_path / "missing")
    with pytest.raises(ServerError, match="fetch_llamacpp"):
        server.chat(MESSAGES)


def test_missing_weights_is_actionable(tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    server = LlamaServer(model_path=tmp_path / "absent.gguf", server_bin=binary)
    with pytest.raises(ServerError, match="pull"):
        server.chat(MESSAGES)


def test_empty_messages_rejected(tmp_path):
    server = LlamaServer(model_path=tmp_path / "m.gguf")
    with pytest.raises(ServerError, match="empty message list"):
        server.chat([])


def test_stop_without_start_is_safe(tmp_path):
    LlamaServer(model_path=tmp_path / "m.gguf").stop()


def test_free_port_is_usable():
    port = free_port()
    assert 1024 < port < 65536
