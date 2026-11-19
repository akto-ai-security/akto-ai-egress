"""File guardrails: find the files in an AI request and check them through data-ingestion.

Files come from two places:
  * multipart/form-data uploads (OpenAI and Anthropic /v1/files): every part with a filename;
  * base64 content inside a JSON body:
      - data URLs (OpenAI chat image_url and file.file_data, Responses input_image and input_file),
      - Anthropic {"type": "base64", "media_type", "data"} sources (image and document blocks),
      - Gemini inline_data / inlineData,
      - Bedrock Converse {"format", "source": {"bytes"}} (image and document blocks).

The files go as multipart to data-ingestion-service /api/validate/file, which forwards them to
guardrails-service /api/validate/file. guardrails-service picks an extractor (OCR for images,
text for documents) by file extension, so every file is sent with one.

Agents re-send their whole history on every turn, so the same files come back every turn. The proxy
keeps no state: every request's files are sent, and guardrails-service remembers the ones it already
inspected and allowed (in its Redis, keyed on the bytes and the policies that apply), so a re-sent file
is answered without being extracted or scanned again, and a policy change applies at once.

Each file's outcome is recorded per request (see check_files): allowed, uninspected, blocked, not-reached
or unchecked. The text guardrails - and the traffic Akto ingests for Traces - get a placeholder in place of
each base64 file, naming it and carrying that outcome; scanning base64 as text finds nothing and can push
the payload over data-ingestion's size limits. restore_files puts the files back into a masked payload
before it is merged into the request.
"""
import base64
import binascii
import hashlib
import json
import mimetypes
import os
import re
from email.parser import BytesParser
from email.policy import HTTP
from typing import NamedTuple, Optional

import requests

# On unless AKTO_FILE_GUARDRAILS=false. Only requests that carry a file make a call.
ENABLED = (os.getenv("AKTO_FILE_GUARDRAILS") or "true").lower() != "false"
MAX_BYTES = int(os.getenv("AKTO_FILE_MAX_BYTES") or 20 * 1024 * 1024)
TIMEOUT_SECONDS = float(os.getenv("AKTO_FILE_TIMEOUT") or 30)
# true (default): an unreachable or failing check lets the request through, like the text guardrails.
FAIL_OPEN = (os.getenv("AKTO_FILE_FAIL_OPEN") or "true").lower() != "false"
# guardrails-service checks at most FILE_VALIDATE_MAX_FILES (default 5) files per call.
BATCH_SIZE = 5

DATA_URL_RE = re.compile(r"^data:([\w.+-]+/[\w.+-]+)?(?:;[^,;]*)*;base64,", re.IGNORECASE)
MAGIC = [
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF8", ".gif"),
    (b"%PDF-", ".pdf"),
]


class File(NamedTuple):
    name: str
    mime: str
    data: bytes
    sha256: str
    raw: Optional[str]  # the base64 text as it appears in a JSON body; None for multipart parts


# Per-file outcomes check_files records. A verdict covers a whole batch (up to BATCH_SIZE files) and doesn't
# name the file that failed, so every file in a blocked batch is "blocked".
ALLOWED, UNINSPECTED, BLOCKED, NOT_REACHED, UNCHECKED = "allowed", "uninspected", "blocked", "not-reached", "unchecked"
_UNSAFE_NAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


def placeholder(f: File, checks: Optional[dict] = None) -> str:
    """What stands in for a base64 file in text payloads: its hash, a JSON-safe name and its check outcome."""
    name = _UNSAFE_NAME_CHARS.sub("_", f.name)[:64]
    return f"[akto-file sha256:{f.sha256[:16]} name={name} check={(checks or {}).get(f.sha256, UNCHECKED)}]"


class FileCheckUnavailable(Exception):
    pass


def _sniff_extension(data: bytes) -> str:
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return next((ext for magic, ext in MAGIC if data.startswith(magic)), "")


def _make_file(name: Optional[str], mime: Optional[str], data: bytes, raw: Optional[str], index: int) -> File:
    """Give the file a name with an extension: the caller's name if it has one, else the
    content's magic bytes, else the declared mime type."""
    name = os.path.basename(name or "") or f"attachment-{index}"
    if not os.path.splitext(name)[1]:
        name += _sniff_extension(data) or (mimetypes.guess_extension(mime or "") or "")
    return File(name, mime or "application/octet-stream", data, hashlib.sha256(data).hexdigest(), raw)


def _b64decode(value) -> Optional[bytes]:
    if not isinstance(value, str):
        return None
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None


def _from_json(node, found: list, filename: Optional[str] = None) -> None:
    """Collect (name, mime, bytes, base64 text) for every base64 file inside a decoded JSON body."""
    if isinstance(node, str):
        match = DATA_URL_RE.match(node)
        data = _b64decode(node[match.end():]) if match else None
        if data is not None:
            found.append((filename, match.group(1), data, node[match.end():]))
        return
    if isinstance(node, list):
        for item in node:
            _from_json(item, found, filename)
        return
    if not isinstance(node, dict):
        return

    # Anthropic: {"type": "base64", "media_type": "image/png", "data": "..."}
    if node.get("type") == "base64" and "data" in node:
        data = _b64decode(node["data"])
        if data is not None:
            found.append((filename, node.get("media_type"), data, node["data"]))
        return
    # Gemini: {"inline_data": {"mime_type": "...", "data": "..."}} (camelCase in the REST API)
    inline = node.get("inline_data") or node.get("inlineData")
    if isinstance(inline, dict):
        data = _b64decode(inline.get("data"))
        if data is not None:
            found.append((filename, inline.get("mime_type") or inline.get("mimeType"), data, inline["data"]))
        return
    # Bedrock Converse: {"format": "png", "name": "...", "source": {"bytes": "..."}}
    source = node.get("source")
    if isinstance(node.get("format"), str) and isinstance(source, dict) and "bytes" in source:
        data = _b64decode(source["bytes"])
        if data is not None:
            name = node.get("name") or filename or "attachment"
            found.append((f"{name}.{node['format']}", None, data, source["bytes"]))
        return

    # OpenAI keeps the name next to the data URL: {"filename": "a.pdf", "file_data": "data:..."}
    name = node.get("filename") if isinstance(node.get("filename"), str) else filename
    for value in node.values():
        _from_json(value, found, name)


def _from_multipart(body: bytes, content_type: str) -> list:
    message = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode("latin-1") + b"\r\n\r\n" + body
    )
    if not message.is_multipart():
        return []
    found = []
    for part in message.iter_parts():
        name = part.get_filename()
        if name is None:
            continue
        mime = part.get_content_type() if part.get("Content-Type") else None
        found.append((name, mime, part.get_payload(decode=True) or b"", None))
    return found


def find_files(body: bytes, content_type: str) -> list:
    """Every file in a request body."""
    if not body:
        return []
    if (content_type or "").lower().startswith("multipart/form-data"):
        found = _from_multipart(body, content_type)
    else:
        try:
            parsed = json.loads(body)
        except ValueError:
            return []
        found = []
        _from_json(parsed, found)
    return [_make_file(name, mime, data, raw, i) for i, (name, mime, data, raw) in enumerate(found, 1)]


def _validate(batch: list, fields: dict, url: str, token: Optional[str]) -> dict:
    headers = {"Authorization": token} if token else {}
    files = [("file", (f.name, f.data, f.mime)) for f in batch]
    try:
        r = requests.post(url, data=fields, files=files, headers=headers, timeout=TIMEOUT_SECONDS)
        r.raise_for_status()
        verdict = r.json()
    except (requests.RequestException, ValueError) as e:
        raise FileCheckUnavailable(str(e)) from e
    # data-ingestion answers allowed=true with failOpen=true when guardrails-service failed.
    if not isinstance(verdict, dict) or "allowed" not in verdict or verdict.get("failOpen"):
        raise FileCheckUnavailable(f"no verdict from file guardrails: {verdict!r}"[:300])
    return verdict


def find_request_files(body: bytes, content_type: str) -> list:
    """The files to check in a request; none when file guardrails are off or the body can't be read."""
    if not ENABLED:
        return []
    try:
        return find_files(body, content_type)
    except Exception as e:  # a body we can't parse must not break the request
        print(f"[AKTO] could not read files from request, not checked: {e}")
        return []


def check_files(files: list, fields: dict, url: str, token: Optional[str], checks: Optional[dict] = None) -> Optional[dict]:
    """Check the files of a request. Returns {"reason", "metadata"} to block it, else None.

    checks, when given, receives each file's outcome by sha256 (see the constants above)."""
    checks = {} if checks is None else checks
    for f in files:
        checks[f.sha256] = UNCHECKED
    pending = {}
    for f in files:
        if len(f.data) > MAX_BYTES:
            print(f"[AKTO] file {f.name} is {len(f.data)} bytes, over AKTO_FILE_MAX_BYTES; not checked")
        else:
            pending.setdefault(f.sha256, f)  # the same file twice in one request is sent once
    if not pending:
        return None

    files = list(pending.values())
    print(f"[AKTO] checking {len(files)} file(s): {[f.name for f in files]}")
    try:
        for start in range(0, len(files), BATCH_SIZE):
            batch = files[start:start + BATCH_SIZE]
            verdict = _validate(batch, fields, url, token)
            if verdict.get("allowed") is False or (verdict.get("behaviour") or "").lower() == "block":
                checks.update({f.sha256: BLOCKED for f in batch})
                checks.update({f.sha256: NOT_REACHED for f in files[start + BATCH_SIZE:]})
                return {
                    "reason": verdict.get("reason") or "File blocked by Akto guardrails",
                    "metadata": verdict.get("metadata") or {},
                }
            if verdict.get("uninspected"):
                print(f"[AKTO] files allowed without inspection: {[f.name for f in batch]}")
            checks.update({f.sha256: UNINSPECTED if verdict.get("uninspected") else ALLOWED for f in batch})
    except FileCheckUnavailable as e:
        if FAIL_OPEN:
            print(f"[AKTO] file guardrails call failed, allowing request: {e}")
            return None
        print(f"[AKTO] file guardrails call failed, blocking request: {e}")
        return {"reason": "File guardrails unavailable", "metadata": {}}
    return None


def hide_files(text: str, files: list, checks: Optional[dict] = None) -> str:
    """Swap each base64 file in a JSON text payload for its placeholder."""
    for f in files:
        if f.raw:
            text = text.replace(f.raw, placeholder(f, checks))
    return text


def restore_files(text: str, files: list, checks: Optional[dict] = None) -> str:
    """Put the base64 files back into a payload that hide_files produced (with the same checks)."""
    for f in files:
        if not f.raw:
            continue
        marker = placeholder(f, checks)
        if marker not in text:
            print(f"[AKTO] masked payload lost file {f.name}; it will be missing from the request")
        text = text.replace(marker, f.raw)
    return text
