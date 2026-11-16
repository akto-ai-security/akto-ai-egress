import base64
import importlib
import json
import os
import sys

import pytest
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import file_guardrails as fg  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) + b"\r\n--not-a-boundary\r\n"
PDF = b"%PDF-1.7\n\xe2\xe3\xcf\xd3\nbody"
PNG_B64 = base64.b64encode(PNG).decode()
PDF_B64 = base64.b64encode(PDF).decode()
URL = "https://akto.example/api/validate/file"
FIELDS = {"path": "/v1/messages", "contextSource": "AGENTIC"}


def files_in(body: dict):
    return fg.find_files(json.dumps(body).encode(), "application/json")


# ---------- finding files ----------

@pytest.mark.parametrize("body, expected", [
    pytest.param(  # OpenAI chat image
        {"messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG_B64}"}}]}]},
        [("attachment-1.png", "image/png", PNG)], id="openai-chat-image"),
    pytest.param(  # OpenAI chat file, name next to the data URL
        {"messages": [{"role": "user", "content": [
            {"type": "file", "file": {"filename": "report.pdf", "file_data": f"data:application/pdf;base64,{PDF_B64}"}}]}]},
        [("report.pdf", "application/pdf", PDF)], id="openai-chat-file"),
    pytest.param(  # OpenAI Responses input_image
        {"input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"}]}]},
        [("attachment-1.png", "image/png", PNG)], id="openai-responses-image"),
    pytest.param(  # Anthropic image and document blocks
        {"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG_B64}},
            {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF_B64}}]}]},
        [("attachment-1.png", "image/png", PNG), ("attachment-2.pdf", "application/pdf", PDF)], id="anthropic"),
    pytest.param(  # Gemini REST (camelCase)
        {"contents": [{"parts": [{"inlineData": {"mimeType": "image/png", "data": PNG_B64}}]}]},
        [("attachment-1.png", "image/png", PNG)], id="gemini"),
    pytest.param(  # Bedrock Converse image and document
        {"messages": [{"role": "user", "content": [
            {"image": {"format": "png", "source": {"bytes": PNG_B64}}},
            {"document": {"format": "pdf", "name": "contract", "source": {"bytes": PDF_B64}}}]}]},
        [("attachment.png", "application/octet-stream", PNG), ("contract.pdf", "application/octet-stream", PDF)],
        id="bedrock"),
])
def test_finds_base64_files_in_json(body, expected):
    assert [tuple(f) for f in files_in(body)] == expected


def test_extension_comes_from_the_bytes_when_the_name_has_none():
    body = {"messages": [{"content": [{"type": "image", "source": {
        "type": "base64", "media_type": "image/jpeg", "data": PNG_B64}}]}]}
    assert files_in(body)[0].name == "attachment-1.png"


@pytest.mark.parametrize("body", [
    {"messages": [{"role": "user", "content": "hello"}]},
    {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}]},
    {"messages": [{"content": [{"type": "image", "source": {"type": "base64", "data": "not base64!"}}]}]},
    {"messages": [{"content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}]}]},
])
def test_ignores_text_urls_and_bad_base64(body):
    assert files_in(body) == []


def test_ignores_non_json_and_empty_bodies():
    assert fg.find_files(b"not json", "application/json") == []
    assert fg.find_files(b"", "application/json") == []


def test_finds_multipart_upload_with_binary_content():
    boundary = "----akto"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nassistants\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"scan.png\"\r\n"
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + PNG + f"\r\n--{boundary}--\r\n".encode()
    files = fg.find_files(body, f"multipart/form-data; boundary={boundary}")
    assert [tuple(f) for f in files] == [("scan.png", "image/png", PNG)]


# ---------- checking files ----------

class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class Calls(list):
    """The validate/file calls made; `replies` are the verdicts to return, in order."""
    replies: list


@pytest.fixture
def calls(monkeypatch):
    monkeypatch.setattr(fg, "ENABLED", True)
    monkeypatch.setattr(fg, "FAIL_OPEN", True)
    recorded = Calls()
    recorded.replies = []

    def post(url, data=None, files=None, headers=None, timeout=None, **_):
        recorded.append({"url": url, "data": data, "files": files, "headers": headers})
        reply = recorded.replies.pop(0) if recorded.replies else {"allowed": True}
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, FakeResponse) else FakeResponse(reply)

    monkeypatch.setattr(fg.requests, "post", post)
    return recorded


def anthropic_body(*blobs):
    content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                            "data": base64.b64encode(b).decode()}} for b in blobs]
    return json.dumps({"messages": [{"role": "user", "content": content}]}).encode()


def check(body, token="jwt"):
    return fg.check_request(body, "application/json", FIELDS, URL, token)


def test_disabled_makes_no_call(calls, monkeypatch):
    monkeypatch.setattr(fg, "ENABLED", False)
    assert check(anthropic_body(PNG)) is None
    assert calls == []


def test_no_files_makes_no_call(calls):
    assert check(b'{"messages": [{"role": "user", "content": "hi"}]}') is None
    assert calls == []


def test_sends_files_and_fields_as_multipart_with_bare_token(calls):
    assert check(anthropic_body(PNG)) is None
    (call,) = calls
    assert call["url"] == URL
    assert call["data"] == FIELDS
    assert call["files"] == [("file", ("attachment-1.png", PNG, "image/png"))]
    assert call["headers"] == {"Authorization": "jwt"}


def test_no_token_sends_no_authorization(calls):
    check(anthropic_body(PNG), token=None)
    assert calls[0]["headers"] == {}


def test_blocked_verdict_blocks_with_reason(calls):
    calls.replies = [{"allowed": False, "reason": "prompt injection in image", "metadata": {"policy": "p1"}}]
    assert check(anthropic_body(PNG)) == {"reason": "prompt injection in image", "metadata": {"policy": "p1"}}


def test_behaviour_block_blocks_even_if_allowed(calls):
    calls.replies = [{"allowed": True, "behaviour": "BLOCK"}]
    assert check(anthropic_body(PNG))["reason"] == "File blocked by Akto guardrails"


def test_batches_of_five_and_stops_at_first_block(calls):
    calls.replies = [{"allowed": True}, {"allowed": False, "reason": "r"}, {"allowed": True}]
    blobs = [PNG + bytes([i]) for i in range(12)]
    assert check(anthropic_body(*blobs))["reason"] == "r"
    assert [len(c["files"]) for c in calls] == [5, 5]


def test_oversized_files_are_skipped(calls, monkeypatch):
    monkeypatch.setattr(fg, "MAX_BYTES", len(PNG) - 1)
    assert check(anthropic_body(PNG)) is None
    assert calls == []


@pytest.mark.parametrize("reply", [
    requests.ConnectionError("refused"),
    requests.Timeout("timed out"),
    FakeResponse({}, status=502),
    FakeResponse({"allowed": True, "failOpen": True}),  # data-ingestion's answer when guardrails-service failed
    FakeResponse({"unexpected": "shape"}),
])
def test_unavailable_check_follows_fail_open(calls, monkeypatch, reply):
    calls.replies = [reply]
    assert check(anthropic_body(PNG)) is None

    monkeypatch.setattr(fg, "FAIL_OPEN", False)
    calls.replies = [reply]
    assert check(anthropic_body(PNG)) == {"reason": "File guardrails unavailable", "metadata": {}}


def test_unparseable_body_is_not_checked(calls, monkeypatch):
    def boom(*_):
        raise RuntimeError("bad body")
    monkeypatch.setattr(fg, "find_files", boom)
    assert check(anthropic_body(PNG)) is None
    assert calls == []


# ---------- the mitmproxy addon ----------

@pytest.fixture
def addon(monkeypatch):
    pytest.importorskip("mitmproxy")
    monkeypatch.setenv("AKTO_URL", "https://akto.example")
    monkeypatch.setenv("AKTO_TOKEN", "jwt")
    import akto_guardrails
    return importlib.reload(akto_guardrails)


def anthropic_flow(body: bytes):
    from mitmproxy.test import tflow
    flow = tflow.tflow()
    flow.request.host = "api.anthropic.com"
    flow.request.path = "/v1/messages"
    flow.request.method = "POST"
    flow.request.headers["content-type"] = "application/json"
    flow.request.content = body
    return flow


def test_addon_blocks_request_on_blocked_file_without_text_check(addon, calls):
    calls.replies = [{"allowed": False, "reason": "injection in image"}]
    flow = anthropic_flow(anthropic_body(PNG))
    addon.AktoGuardrailsAddon().request(flow)

    assert flow.response.status_code == 403
    assert flow.response.headers["X-Akto-Guardrails-Decision"] == "blocked"
    assert json.loads(flow.response.content)["error"] == "injection in image"
    assert [c["url"] for c in calls] == ["https://akto.example/api/validate/file"]
    assert calls[0]["data"]["path"] == "/v1/messages" and calls[0]["data"]["contextSource"] == "AGENTIC"


def test_addon_runs_text_check_after_allowed_file(addon, calls, monkeypatch):
    text_calls = []
    monkeypatch.setattr(addon, "post_to_akto", lambda params, payload: text_calls.append(payload) or
                        {"data": {"guardrailsResult": {"Allowed": True}}})
    flow = anthropic_flow(anthropic_body(PNG))
    addon.AktoGuardrailsAddon().request(flow)

    assert flow.response is None
    assert len(calls) == 1 and len(text_calls) == 1
    assert text_calls[0]["path"] == "/v1/messages" and text_calls[0]["tag"] == addon.TAGS
